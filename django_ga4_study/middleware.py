import secrets


class GA4StudyUserIdMiddleware:
    """Ensures a pseudonymous, non-PII user id is present in session."""

    SESSION_KEY = 'ga4_study_user_id'

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        sid = request.session.get(self.SESSION_KEY)
        if not sid:
            # compact, pseudonymous id
            sid = 'u_' + secrets.token_hex(6)  # ~12 chars
            request.session[self.SESSION_KEY] = sid
        request.ga4_study_user_id = sid
        return self.get_response(request)
