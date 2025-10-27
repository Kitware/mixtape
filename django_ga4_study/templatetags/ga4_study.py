from django import template
from django.conf import settings

register = template.Library()


@register.inclusion_tag('django_ga4_study/ga4_head.html', takes_context=True)
def ga4_head(context):
    request = context.get('request')
    user_id = getattr(request, 'ga4_study_user_id', None) or getattr(
        getattr(request, 'session', None), 'get', lambda *_: None
    )('ga4_study_user_id')
    return {
        'GA_MEASUREMENT_ID': getattr(settings, 'GA_MEASUREMENT_ID', ''),
        'GA_INSTANCE_ID': getattr(settings, 'GA_INSTANCE_ID', 'dev-instance'),
        'GA_DEBUG': bool(
            getattr(settings, 'GA_DEBUG', settings.DEBUG if hasattr(settings, 'DEBUG') else False)
        ),
        'GA_SAMPLE': float(getattr(settings, 'GA_SAMPLE', 1.0)),  # 0.0-1.0 sampling
        'STUDY_USER_ID': user_id or 'anon',
    }
