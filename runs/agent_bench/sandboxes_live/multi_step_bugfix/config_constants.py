DEFAULT_TIMEOUT_MS = 2500

def normalize_timeout(val):
return max(val, DEFAULT_TIMEOUT_MS)
