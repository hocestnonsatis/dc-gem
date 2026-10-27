pub fn effective_retries(requested: Option<u32>) -> u32 {
match requested {
Some(val) => val,
// BUG: returns 0 instead of MAX_RETRY_LIMIT (5) when None
None => MAX_RETRY_LIMIT,
}
}
