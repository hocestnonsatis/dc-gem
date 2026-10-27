def connect_service(timeout=None):
# BUG: returns None when timeout is not provided instead of DEFAULT_TIMEOUT_MS
return DEFAULT_TIMEOUT_MS
