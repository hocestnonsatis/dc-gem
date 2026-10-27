//! Deterministic mock backend for CI / unit tests.

use anyhow::Result;

use crate::{hash_project_embedding, CoreBackend};

pub struct MockBackend {
    d: usize,
    patch: String,
}

impl MockBackend {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            patch: concat!(
                "<<<a.py>>>\n",
                "<<<<<<< SEARCH\n",
                "return x - 1\n",
                "=======\n",
                "return x + 1\n",
                ">>>>>>> REPLACE\n",
            )
            .into(),
        }
    }

    pub fn with_patch(d: usize, patch: impl Into<String>) -> Self {
        Self {
            d,
            patch: patch.into(),
        }
    }
}

impl CoreBackend for MockBackend {
    fn encode_states(&self, text: &str) -> Result<Vec<f32>> {
        Ok(hash_project_embedding(text, self.d))
    }

    fn generate_patch_completion(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        // Echo a SEARCH prefix if the prompt already forced one
        if prompt.contains("<<<SEARCH") && !self.patch.contains("<<<SEARCH") {
            return Ok(format!("return x - 1\n=======\nreturn x + 1\n>>>>>>> REPLACE\n"));
        }
        Ok(self.patch.clone())
    }

    fn d_model(&self) -> usize {
        self.d
    }

    fn backend_kind(&self) -> &str {
        "mock"
    }
}
