"""Durable Engui Identity Edit lane. Ordinary jobs, never RunPod Batch Jobs.

A committed outbox entry owns each paid attempt. Network retries reuse that
attempt's key. No ComfyUI queue, credentials, or filesystem paths cross the bridge.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import threading
import time
import uuid
from urllib.parse import urlsplit

import requests
from filelock import FileLock, Timeout
from PIL import Image, ImageOps

from .. import config as cfg
from ..extensions import db
from ..models import EnguiDatasetJob, FaceDataset, FaceDatasetImage

ENGINE = 'engui_krea'
TERMINAL = ('succeeded', 'failed', 'cancelled')
_worker = None
_start_lock = threading.Lock()


class BridgeError(ValueError):
    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def connection():
    base = str(cfg.get('engui.url') or '').rstrip('/')
    workspace = str(cfg.get('engui.workspace_id') or '')
    token = os.environ.get('LDS_ENGUI_TOKEN', '')
    parsed = urlsplit(base)
    if (parsed.scheme not in ('https', 'http') or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise BridgeError('Set the Engui server origin in Settings (without an API path).')
    if parsed.scheme != 'https' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise BridgeError('Remote Engui connections require HTTPS.')
    try:
        uuid.UUID(workspace)
    except ValueError:
        raise BridgeError('Set the Engui workspace ID in Settings.') from None
    if not token:
        raise BridgeError('Set LDS_ENGUI_TOKEN in Settings or the server environment.')
    return base, workspace, token


class Client:
    def __init__(self, base=None, workspace=None):
        current_base, current_workspace, self.token = connection()
        self.base = base or current_base
        self.workspace = workspace or current_workspace
        if (self.base, self.workspace) != (current_base, current_workspace):
            raise BridgeError('Pending jobs belong to another Engui connection. Restore its settings.',
                              retryable=True)

    def call(self, action, **payload):
        try:
            response = requests.post(
                self.base + '/api/dataset-studio',
                json={'action': action, **payload},
                headers={'Authorization': 'Bearer ' + self.token,
                         'X-Workspace-Id': self.workspace},
                timeout=(2, 5) if action == 'capabilities' else (10, 45), allow_redirects=False)
        except requests.RequestException:
            raise BridgeError('Engui connection interrupted; the same job will be checked again.',
                              retryable=True) from None
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code != 200:
            code = str(data.get('code') or 'HTTP_' + str(response.status_code))[:80]
            raise BridgeError('Engui request failed (' + code + ').',
                              retryable=(response.status_code >= 500 or response.status_code
                                         in (408, 409, 429) and code in
                                         ('REQUEST_FAILED', 'RATE_LIMITED', 'NOT_READY')))
        if not isinstance(data, dict):
            raise BridgeError('Invalid Engui response.', retryable=True)
        return data

    def transfer(self, method, url, *, data=None, mime=None, limit=25 * 1024 * 1024):
        # Signed links are exact-file capabilities. Never forward the long-lived
        # bearer to redirects or to an origin supplied by a provider result.
        if (urlsplit(url).scheme, urlsplit(url).netloc) != (
                urlsplit(self.base).scheme, urlsplit(self.base).netloc):
            raise BridgeError('Engui returned a file link for a different server.')
        try:
            with requests.request(method, url, data=data,
                                  headers={'Content-Type': mime} if mime else {},
                                  timeout=(10, 60), allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    raise BridgeError('File transfer interrupted; a fresh link will be requested.',
                                      retryable=True)
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > limit:
                        raise BridgeError('Engui image exceeds the transfer limit.')
                    chunks.append(chunk)
                return b''.join(chunks)
        except requests.RequestException:
            raise BridgeError('File transfer interrupted; it will be resumed.',
                              retryable=True) from None


def preflight():
    ready = Client().call('capabilities')
    if not ready.get('ready') or ready.get('workflow') != 'krea2-identity-v1.2':
        raise BridgeError('Engui Krea Identity Edit is not ready on this server.')
    return ready


def probe():
    try:
        preflight()
        return {'ok': True, 'detail': 'Engui Krea Identity Edit is ready.'}
    except (BridgeError, ValueError):
        return {'ok': False, 'detail': 'Configure or reconnect Engui in Settings.'}


def _root():
    root = Path(cfg.data_dir()) / 'engui-jobs'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_job(user_id, ds, img, variation):
    from . import face_dataset_service as svc
    from . import krea_edit_helper as krea
    base, workspace, _ = connection()
    key = str(uuid.uuid4())
    # Pin the reference bytes, not a live reference pathname that can be recropped.
    with Image.open(svc._ref_path(ds)) as original:
        out = io.BytesIO()
        ImageOps.exif_transpose(original).convert('RGB').save(out, 'PNG')
    raw = out.getvalue()
    if len(raw) > 25 * 1024 * 1024:
        raise BridgeError('Reference exceeds the 25 MiB upload limit.')
    sha = hashlib.sha256(raw).hexdigest()
    ref = _root() / (sha + '.png')
    if not ref.exists():
        svc.write_image_atomic(str(ref), raw)
    aspect = svc.aspect_for_label(variation.get('label'), variation.get('framing'))
    # The upstream card's framing is retained; ~1 MP, on the worker's 64px grid.
    a, b = (float(n) for n in aspect.split(':'))
    width = max(512, min(2048, round((1048576 * a / b) ** .5 / 64) * 64))
    height = max(512, min(2048, round((1048576 * b / a) ** .5 / 64) * 64))
    prompt = svc.wrap_variation_krea(
        variation['prompt'], nsfw=False, framing=variation.get('framing'),
        suffix=svc.dataset_prompt_suffix(ds, variation.get('framing')),
        subject_type=svc.subject_type_of(ds), label=variation.get('label') or '')
    if len(prompt) > 16000:
        raise BridgeError('The composed Studio prompt exceeds the Engui limit.')
    controls = {'grounding_px': krea.grounding_px(), 'ref_boost': krea._ref_boost(),
                'identity_strength': krea._identity_strength()}
    payload = {'requestKey': key, 'prompt': prompt, 'seed': secrets.randbelow(2147483648),
               'width': width, 'height': height, 'steps': krea._steps(),
               'identity': controls, 'loras': [],
               'provenance': {'datasetId': str(ds.id), 'candidateId': str(img.id),
                              'cardLabel': variation.get('label') or '',
                              'subjectType': svc.subject_type_of(ds)}}
    job = EnguiDatasetJob(id=key, dataset_id=ds.id, image_id=img.id, user_id=str(user_id),
                          base_url=base, workspace_id=workspace, reference_hash=sha,
                          payload_json=json.dumps(payload), state='queued')
    img.job_id = key
    img.klein_model = ENGINE
    img.generation_meta = json.dumps({'engine': ENGINE, 'seed': payload['seed'],
                                      'steps': payload['steps'], 'aspect': aspect,
                                      'identity': controls, 'model': 'krea2-turbo'})
    db.session.add(job)
    return job


def generate(user_id, dataset_id, variations, multiplier=1):
    from . import face_dataset_service as svc
    svc._guard_not_bank_export(dataset_id)
    ds = svc.get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        raise BridgeError('A dataset reference image is required.')
    mult = int(multiplier)
    if mult < 1 or len(variations) * mult > 100:
        raise BridgeError('Queue between 1 and 100 Engui candidates per request.')
    if any(v.get('nsfw') or svc.is_nsfw_label(v.get('label')) for v in variations):
        raise BridgeError('This remote engine does not accept the local-only catalog.')
    preflight()
    ids = []
    try:
        for variation in variations:
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=ds.id, source='generated', status='pending',
                                       variation_prompt=variation['prompt'],
                                       variation_label=variation.get('label'),
                                       framing=variation.get('framing'), klein_model=ENGINE)
                db.session.add(img)
                db.session.flush()
                _new_job(user_id, ds, img, variation)
                ids.append(img.id)
        # Candidates and paid intents become visible together, before any HTTP.
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    svc._sync_generate_activity(dataset_id)
    return ids


def cancel_job(job_id, user_id=None, *, commit=True):
    """None = not ours, retry = cancellation durable but not confirmed."""
    job = db.session.get(EnguiDatasetJob, job_id)
    if job is None:
        return None
    if user_id is not None and job.user_id != str(user_id):
        return 'retry'
    if job.state in TERMINAL:
        return 'terminal'
    job.cancel_requested = True
    # CAS: a worker may have claimed this job since our SELECT. Never call an
    # unknown submission "cancelled" just because the stale object was queued.
    changed = (EnguiDatasetJob.query.filter_by(id=job_id, state='queued')
               .update({'state': 'cancelled', 'cancel_requested': True},
                       synchronize_session=False))
    if commit:
        db.session.commit()
    return 'cancelled' if changed else 'retry'


def regenerate(user_id, image_id, prompt):
    from . import face_dataset_service as svc
    img = svc._owned_image(user_id, image_id)
    if not img:
        return None
    if svc.is_nsfw_label(img.variation_label):
        raise BridgeError('This remote engine does not accept the local-only catalog.')
    preflight()
    if img.job_id:
        outcome = cancel_job(img.job_id, user_id)
        if outcome == 'retry':
            raise BridgeError('The previous Engui attempt is stopping. Retry after it finishes.')
        if outcome is None and not img.filename:
            raise BridgeError('Stop the previous local generation before switching to Engui.')
    ds = db.session.get(FaceDataset, img.dataset_id)
    old_path = svc._img_path(img) if img.filename else None
    variation = {'prompt': prompt, 'label': img.variation_label, 'framing': img.framing}
    try:
        job = _new_job(user_id, ds, img, variation)
        img.variation_prompt = prompt
        img.filename = None
        svc.caption_origin.stamp(img, None, None)
        svc.caption_origin.stamp(img, None, None, field='caption_short')
        svc._clear_watermark_metadata(img)
        img.status = 'pending'
        img.fail_reason = None
        img.fail_kind = None
        svc._invalidate_image_content_analysis(img)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    if old_path and os.path.isfile(old_path):
        # Preserve the old original in the app's trash, never unlink user data.
        svc.trash.send_to_trash(old_path, context=f'dataset-{ds.id}-regenerate')
    svc._sync_generate_activity(ds.id)
    return job.id


def _finish(job, state, reason=None):
    from . import face_dataset_service as svc
    job.state = state
    job.last_error = reason
    img = db.session.get(FaceDatasetImage, job.image_id)
    if img and img.job_id == job.id and not img.filename:
        if state == 'cancelled':
            svc._finish_cancelled_generation_row(img)
        elif state == 'failed':
            img.status = 'failed'
            img.fail_reason = reason or 'Engui generation failed.'
    db.session.commit()
    svc._sync_generate_activity(job.dataset_id)


def tick():
    """Advance one durable attempt; a process-wide file lock owns dispatch."""
    from . import face_dataset_service as svc
    db.session.remove()
    job = (EnguiDatasetJob.query.filter(~EnguiDatasetJob.state.in_(TERMINAL))
           .order_by(EnguiDatasetJob.created_at, EnguiDatasetJob.id).first())
    if job is None:
        return False
    svc._sync_generate_activity(job.dataset_id)
    img = db.session.get(FaceDatasetImage, job.image_id)
    if not img or img.job_id != job.id:
        job.cancel_requested = True
        db.session.commit()
    try:
        if job.state == 'queued':
            if job.cancel_requested:
                _finish(job, 'cancelled')
                return True
            claimed = (EnguiDatasetJob.query.filter_by(id=job.id, state='queued', cancel_requested=False)
                       .update({'state': 'uploading'}, synchronize_session=False))
            db.session.commit()
            if not claimed:
                return True
            db.session.refresh(job)
        client = Client(job.base_url, job.workspace_id)
        if job.state == 'uploading':
            if job.cancel_requested:
                _finish(job, 'cancelled')
                return True
            raw = (_root() / (job.reference_hash + '.png')).read_bytes()
            upload_id = str(uuid.uuid5(uuid.UUID(job.workspace_id),
                                      hashlib.sha256(client.token.encode()).hexdigest() + ':' + job.reference_hash))
            uploaded = client.call('create_upload', input={
                'requestKey': upload_id, 'filename': 'reference.png', 'mimeType': 'image/png',
                'sha256': job.reference_hash, 'byteLength': len(raw)})
            if uploaded.get('status') != 'ready':
                client.transfer('PUT', uploaded['uploadUrl'], data=raw, mime='image/png')
                uploaded = client.call('get_upload', uploadId=upload_id)
            if uploaded.get('status') != 'ready':
                raise BridgeError('Reference upload is not ready.', retryable=True)
            payload = json.loads(job.payload_json)
            payload['source'] = uploaded['source']
            job.payload_json = json.dumps(payload)
            job.state = 'submitting'
            db.session.commit()
        if job.state == 'submitting':
            # Also replay an interrupted submit before cancellation: its outcome
            # may be unknown. Engui's key binds it to the same single paid attempt.
            result = client.call('submit', input=json.loads(job.payload_json))
            job.remote_job_id = result['jobId']
            job.state = 'running'
            db.session.commit()
        # Observe cancellation from another request after the network call.
        db.session.refresh(job)
        if job.cancel_requested:
            result = client.call('cancel', requestKey=job.id,
                                 cancelKey=str(uuid.uuid5(uuid.UUID(job.id), 'cancel')))
        else:
            result = client.call('status', requestKey=job.id)
        job.last_error = None
        if result.get('isTerminal'):
            if job.cancel_requested:
                _finish(job, 'cancelled')
            elif result.get('state') == 'succeeded':
                descriptor = client.call('artifact', requestKey=job.id)
                raw = client.transfer('GET', descriptor['downloadUrl'])
                if (len(raw) != descriptor['byteLength']
                        or hashlib.sha256(raw).hexdigest() != descriptor['sha256']):
                    raise BridgeError('Original image integrity check failed.', retryable=True)
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                db.session.refresh(job)
                img = db.session.get(FaceDatasetImage, job.image_id)
                if job.cancel_requested or not img or img.job_id != job.id:
                    _finish(job, 'cancelled')
                else:
                    # Keep exact original bytes; no preview rescaling or finishing filter.
                    extension = {'image/png': '.png', 'image/jpeg': '.jpg',
                                 'image/webp': '.webp'}.get(descriptor['mimeType'])
                    if not extension:
                        raise BridgeError('Engui returned an unsupported image format.')
                    name = 'Engui_' + job.id + extension
                    svc.write_image_atomic(str(Path(svc._dataset_dir(job.dataset_id)) / name), raw)
                    img.filename = name
                    _finish(job, 'succeeded')
            else:
                _finish(job, 'failed', 'Engui generation ended: ' + str(result.get('errorCode') or result['state']))
        else:
            db.session.commit()
    except BridgeError as exc:
        db.session.rollback()
        job = db.session.get(EnguiDatasetJob, job.id)
        if not exc.retryable and job.state not in ('submitting', 'running'):
            _finish(job, 'failed', str(exc))
        else:
            # A permanent auth/server error cannot justify another paid attempt.
            # Keep the durable handle and expose the connection error for recovery.
            job.last_error = str(exc)
            db.session.commit()
    return True


def start(app):
    global _worker
    with _start_lock:
        if app.config.get('TESTING') or _worker is not None and _worker.is_alive():
            return

        def loop():
            while True:
                try:
                    with FileLock(str(_root() / 'dispatch.lock'), timeout=0):
                        with app.app_context():
                            try:
                                tick()
                            finally:
                                db.session.remove()
                except Timeout:
                    pass
                except Exception:
                    # Never log signed URLs, payloads, tokens or raw HTTP errors.
                    app.logger.error('Engui queue tick failed; durable attempt retained.')
                time.sleep(3)
        _worker = threading.Thread(target=loop, name='engui-dataset-queue', daemon=True)
        _worker.start()
