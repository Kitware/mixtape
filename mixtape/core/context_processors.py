from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest


def demo_feedback_modal(_request: HttpRequest) -> dict[str, bool]:
    return {'demo_feedback_modal_enabled': getattr(settings, 'DEMO_FEEDBACK_MODAL_ENABLED', False)}
