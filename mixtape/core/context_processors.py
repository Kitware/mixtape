from __future__ import annotations

from django.conf import settings
from django.http import HttpRequest


def mixtape_feedback_modal_enabled(_request: HttpRequest) -> dict[str, bool]:
    return {'mixtape_feedback_modal_enabled': settings.MIXTAPE_FEEDBACK_MODAL_ENABLED}
