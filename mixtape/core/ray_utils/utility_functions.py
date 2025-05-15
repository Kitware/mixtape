from mixtape.core.models.action_mapping import ActionMapping


def get_environment_mapping(environment: str) -> dict:
    try:
        custom_mapping = ActionMapping.objects.get(environment=environment)
    except ActionMapping.DoesNotExist:
        return {}
    return custom_mapping.mapping
