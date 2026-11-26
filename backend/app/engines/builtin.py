"""Built-in local engines and the optional Engui remote engine."""
from .registry import EngineSpec, LOCAL, REMOTE, get, register


def _engui_probe():
    from ..services.engui_remote import probe
    return probe()


def register_builtin():
    # Readiness follows the local daemon; it is not a durable installation check.
    for spec in (
        EngineSpec(id='klein', label='Klein', kind=LOCAL, order=0,
                   counts_as_recommended=True),
        EngineSpec(id='krea', label='Krea 2 Edit', kind=LOCAL, order=1),
        EngineSpec(id='engui_krea', label='Krea 2 Edit · Engui', kind=REMOTE, order=2,
                   remote=True, billable=True, rate_usd_per_image=None,
                   secret='LDS_ENGUI_TOKEN', counts_as_recommended=True,
                   probe=_engui_probe, key_test_target='engui_krea', file_tag='Engui',
                   extra={'supports_reference_edit': False}),
    ):
        if get(spec.id) is None:
            register(spec)
