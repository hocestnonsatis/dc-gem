pub const MAX_RETRY_LIMIT: u32 = 5;

pub fn clamp_retries(requested: u32) -> u32 {
if requested > MAX_RETRY_LIMIT {
MAX_RETRY_LIMIT
} else {
requested
}
}
