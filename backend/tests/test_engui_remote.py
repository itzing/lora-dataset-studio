"""Paid attempts survive restart, uncertain replies and cancellation."""
import hashlib
import io
import json
from pathlib import Path
import uuid
import zipfile

import pytest
from PIL import Image

from app import config as cfg
from app.extensions import db
from app.models import EnguiDatasetJob, FaceDatasetImage
from app.services import engui_remote as remote, face_dataset_service as svc
from test_generate_multi_engine import _dataset_with_ref, _shots


@pytest.fixture
def provider(app, monkeypatch):
    cfg.save_config({'engui': {'url': 'https://bridge.example.test',
                              'workspace_id': str(uuid.uuid4())},
                     'engines': {'enabled': ['engui_krea']}})
    monkeypatch.setenv('LDS_ENGUI_TOKEN', 'test-only-secret')
    calls, accepted = [], {}
    out = io.BytesIO()
    Image.new('RGB', (512, 512), '#337799').save(out, 'PNG')
    original = out.getvalue()
    state = {'terminal': False, 'lose_submit_reply': False}

    def call(_self, action, **payload):
        calls.append((action, payload))
        if action == 'capabilities':
            return {'ready': True, 'workflow': 'krea2-identity-v1.2'}
        if action in ('create_upload', 'get_upload'):
            return {'status': 'ready', 'source': {'kind': 'gallery_asset', 'assetId': str(uuid.uuid4())}}
        if action == 'submit':
            key = payload['input']['requestKey']
            accepted.setdefault(key, {'jobId': str(uuid.uuid4())})
            if state['lose_submit_reply']:
                state['lose_submit_reply'] = False
                raise remote.BridgeError('Lost response', retryable=True)
            return accepted[key]
        if action == 'cancel':
            return {'isTerminal': True, 'state': 'cancelled'}
        if action == 'status':
            return {'isTerminal': state['terminal'],
                    'state': 'succeeded' if state['terminal'] else 'provider_queued'}
        if action == 'artifact':
            return {'downloadUrl': 'https://bridge.example.test/original',
                    'mimeType': 'image/png', 'byteLength': len(original),
                    'sha256': hashlib.sha256(original).hexdigest()}
        raise AssertionError(action)

    monkeypatch.setattr(remote.Client, 'call', call)
    monkeypatch.setattr(remote.Client, 'transfer', lambda *a, **k: original)
    return calls, accepted, state, original


def test_25_candidates_keep_upstream_prompts_and_pin_reference(client, provider):
    dataset_id = _dataset_with_ref(client)
    calls, accepted, _, _ = provider
    response = client.post(f'/api/dataset/{dataset_id}/generate', json={
        'generator': 'engui_krea', 'variations': _shots(25), 'multiplier': 1})
    assert response.status_code == 200, response.json
    assert response.json['created'] == 25
    assert not accepted
    with client.application.app_context():
        jobs = EnguiDatasetJob.query.all()
        assert len(jobs) == 25
        assert len({j.id for j in jobs}) == 25
        for job in jobs:
            row = db.session.get(FaceDatasetImage, job.image_id)
            ds = svc.get_dataset('local', dataset_id)
            payload = json.loads(job.payload_json)
            assert payload['prompt'] == svc.wrap_variation_krea(
                row.variation_prompt, nsfw=False, framing=row.framing,
                suffix=svc.dataset_prompt_suffix(ds, row.framing),
                subject_type=svc.subject_type_of(ds), label=row.variation_label)
            assert Path(remote._root() / (job.reference_hash + '.png')).is_file()
            assert 'test-only-secret' not in job.payload_json
            assert row.job_id == job.id


def test_unknown_submit_replays_same_attempt_then_downloads_original(client, provider):
    dataset_id = _dataset_with_ref(client)
    calls, accepted, state, original = provider
    with client.application.app_context():
        ids = remote.generate('local', dataset_id, _shots(1))
        state['lose_submit_reply'] = True
        remote.tick()
        key = EnguiDatasetJob.query.one().id
        assert EnguiDatasetJob.query.one().state == 'submitting'
        db.session.remove()  # a restarted worker sees only durable state
        remote.tick()
        assert len(accepted) == 1
        assert [p['input']['requestKey'] for a, p in calls if a == 'submit'] == [key, key]
        state['terminal'] = True
        remote.tick()
        row = db.session.get(FaceDatasetImage, ids[0])
        assert Path(svc._img_path(row)).read_bytes() == original
        assert row.status == 'pending'  # review remains the user's decision
        assert EnguiDatasetJob.query.one().state == 'succeeded'


def test_stop_before_dispatch_never_submits(client, provider):
    dataset_id = _dataset_with_ref(client)
    with client.application.app_context():
        remote.generate('local', dataset_id, _shots(3))
        result = svc.cancel_pending('local', dataset_id)
        assert result['cancelled'] == 3
        assert not remote.tick()
        assert not provider[1]


def test_stop_after_lost_reply_cancels_exact_remote_attempt(client, provider):
    dataset_id = _dataset_with_ref(client)
    with client.application.app_context():
        remote.generate('local', dataset_id, _shots(1))
        provider[2]['lose_submit_reply'] = True
        remote.tick()
        job = EnguiDatasetJob.query.one()
        assert remote.cancel_job(job.id, 'local') == 'retry'
        remote.tick()
        assert len(provider[1]) == 1
        assert EnguiDatasetJob.query.one().state == 'cancelled'
        assert FaceDatasetImage.query.filter_by(dataset_id=dataset_id).count() == 0


def test_deleting_candidate_recovers_provider_cancellation(client, provider):
    dataset_id = _dataset_with_ref(client)
    with client.application.app_context():
        ids = remote.generate('local', dataset_id, _shots(1))
        remote.tick()
        db.session.delete(db.session.get(FaceDatasetImage, ids[0]))
        db.session.commit()
        remote.tick()
        assert EnguiDatasetJob.query.one().state == 'cancelled'
        assert len(provider[1]) == 1


def test_retry_preserves_original_and_creates_new_attempt(client, provider):
    dataset_id = _dataset_with_ref(client)
    with client.application.app_context():
        ids = remote.generate('local', dataset_id, _shots(1))
        provider[2]['terminal'] = True
        remote.tick()
        row = db.session.get(FaceDatasetImage, ids[0])
        previous = row.job_id
        row.caption = 'Previous caption'
        row.caption_origin = 'asserted'
        row.caption_short = 'Previous short caption'
        row.caption_short_origin = 'asserted'
        row.watermark_state = 'detected'
        db.session.commit()
        key = svc.regenerate_image('local', row.id)
        assert key != previous
        assert EnguiDatasetJob.query.count() == 2
        assert db.session.get(FaceDatasetImage, row.id).filename is None
        assert row.caption is None and row.caption_short is None
        assert row.caption_origin is None and row.caption_short_origin is None
        assert row.watermark_state is None
        progress = svc.dataset_payload('local', dataset_id)['images'][0]['remote_progress']
        assert progress['state'] == 'queued' and progress['label'] == 'Queued'
        remote.tick()
        assert len(provider[1]) == 2


def test_exact_export_is_25_pairs_without_extra_reference(client, provider):
    dataset_id = _dataset_with_ref(client)
    with client.application.app_context():
        for n in range(25):
            name = f'fixture-{n}.png'
            Path(svc._dataset_dir(dataset_id), name).write_bytes(provider[3])
            db.session.add(FaceDatasetImage(dataset_id=dataset_id, source='generated',
                                          status='keep', filename=name, caption='A portrait.'))
        db.session.commit()
        out = io.BytesIO()
        svc.write_export_zip('local', dataset_id, out, exact_25=True)
        with zipfile.ZipFile(io.BytesIO(out.getvalue())) as archive:
            assert len([n for n in archive.namelist() if n.endswith('.png')]) == 25
            assert len([n for n in archive.namelist() if n.endswith('.txt')]) == 25
            assert not any('_ref.' in n for n in archive.namelist())
        row = FaceDatasetImage.query.filter_by(dataset_id=dataset_id).first()
        row.caption = ''
        db.session.commit()
        with pytest.raises(ValueError, match='Caption'):
            svc.write_export_zip('local', dataset_id, io.BytesIO(), exact_25=True)


def test_tall_drawn_reference_survives_upload_and_remote_snapshot(client, provider):
    """The user's 400x1024 reference lost its head to a centered square."""
    from app.services.face_variations import select_preset, presets_for
    dsid = client.post('/api/dataset/create', json={
        'name': 'Drawn character', 'trigger_word': 'drawn_ref', 'subject_type': 'anime'}).json['id']
    im = Image.new('RGB', (400, 1024), 'green')
    im.paste('red', (0, 0, 400, 140))  # head, outside the old centered square
    im.paste('blue', (0, 900, 400, 1024))  # feet, also outside it
    raw = io.BytesIO(); im.save(raw, 'PNG')
    uploaded = client.post(f'/api/dataset/{dsid}/ref', data={
        'file': (io.BytesIO(raw.getvalue()), 'character.png')}, content_type='multipart/form-data')
    assert uploaded.status_code == 200
    with client.application.app_context():
        shot = select_preset(next(iter(presets_for('anime'))), 'anime')[0]
        remote.generate('local', dsid, [shot])
        job = EnguiDatasetJob.query.one()
        pinned = Image.open(remote._root() / (job.reference_hash + '.png')).convert('RGB')
        assert pinned.size == (400, 1024)
        assert pinned.getpixel((200, 50))[0] > 220
        assert pinned.getpixel((200, 975))[2] > 220
        prompt = json.loads(job.payload_json)['prompt']
        assert 'Professional realistic photograph' not in prompt
        assert 'Create a new photograph' not in prompt
        assert 'same signature outfit' in prompt
        assert 'same drawn anime art style' in prompt
        assert not provider[1]  # inspection does not itself dispatch a paid job


def test_explicit_manual_crop_remains_the_remote_reference(client, provider):
    dsid = _dataset_with_ref(client)
    with client.application.app_context():
        ds = svc.get_dataset('local', dsid)
        original = Path(svc._ref_crop_source_path(ds)).read_bytes()
        assert svc.crop_reference('local', dsid, 0, 0, 24, 40)
        remote.generate('local', dsid, _shots(1))
        job = EnguiDatasetJob.query.one()
        assert Image.open(remote._root() / (job.reference_hash + '.png')).size == (24, 40)
        assert Path(svc._ref_crop_source_path(ds)).read_bytes() == original
