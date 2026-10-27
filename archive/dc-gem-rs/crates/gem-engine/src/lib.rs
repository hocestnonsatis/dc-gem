//! Cross-platform model backends for DC-GEM.

mod gguf;
mod mock;

pub use gguf::{format_qwen35_chat, hash_project_embedding, GgufBackend, GgufConfig};
pub use mock::MockBackend;

use anyhow::Result;

/// Core embedding + patch completion trait.
pub trait CoreBackend: Send + Sync {
    fn encode_states(&self, text: &str) -> Result<Vec<f32>>;
    fn generate_patch_completion(&self, prompt: &str, max_tokens: usize) -> Result<String>;
    fn d_model(&self) -> usize;
    fn backend_kind(&self) -> &str;
}

/// Resolve which GPU offload flag story to use based on Cargo features.
pub fn accel_label() -> &'static str {
    #[cfg(feature = "rocm")]
    {
        return "rocm";
    }
    #[cfg(all(feature = "cuda", not(feature = "rocm")))]
    {
        return "cuda";
    }
    #[cfg(all(feature = "metal", not(any(feature = "rocm", feature = "cuda"))))]
    {
        return "metal";
    }
    #[cfg(all(
        feature = "vulkan",
        not(any(feature = "rocm", feature = "cuda", feature = "metal"))
    ))]
    {
        return "vulkan";
    }
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan"
    )))]
    {
        "cpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_backend_encode_and_generate() {
        let b = MockBackend::new(64);
        let v = b.encode_states("def foo():\n    return 1\n").unwrap();
        assert_eq!(v.len(), 64);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
        let out = b
            .generate_patch_completion("fix the bug\n", 64)
            .unwrap();
        assert!(out.contains("SEARCH") || out.contains("REPLACE") || out.contains("return"));
    }

    #[test]
    fn hash_embed_deterministic() {
        let a = hash_project_embedding("hello world", 32);
        let b = hash_project_embedding("hello world", 32);
        assert_eq!(a, b);
    }

    #[test]
    fn qwen_chat_disables_thinking() {
        let p = format_qwen35_chat(&[("user", "hi")], true, false);
        assert!(p.contains("<think>"));
        assert!(p.contains("</think>"));
        assert!(p.contains("<|im_start|>assistant"));
    }
}
