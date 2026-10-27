//! GGUF / llama-server backend with SEARCH prefix forcing.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use md5::{Digest as _, Md5};
use sha2::Sha256;

use crate::{accel_label, CoreBackend};

pub const DEFAULT_D_MODEL: usize = 4096;

#[derive(Clone, Debug)]
pub struct GgufConfig {
    pub model_path: Option<PathBuf>,
    pub server_url: String,
    pub d_model: usize,
    pub force_search_prefix: bool,
    pub n_gpu_layers: i32,
    pub cli_bin: Option<PathBuf>,
}

impl Default for GgufConfig {
    fn default() -> Self {
        let server_url = env::var("DC_GEM_LLAMA_SERVER_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8765".into());
        Self {
            model_path: resolve_default_gguf(),
            server_url,
            d_model: DEFAULT_D_MODEL,
            force_search_prefix: true,
            n_gpu_layers: if accel_label() == "cpu" { 0 } else { 99 },
            cli_bin: None,
        }
    }
}

fn resolve_default_gguf() -> Option<PathBuf> {
    let candidates = [
        "models/ternary_bonsai_9b/Ternary-Bonsai-2-9B-Q2_0.gguf",
        "models/ternary_bonsai_9b/Ternary-Bonsai-2-9B-PQ2_0.gguf",
        "models/ternary_bonsai_9b/Ternary-Bonsai-2-9B-PTQ1_0.gguf",
        "../models/ternary_bonsai_9b/Ternary-Bonsai-2-9B-Q2_0.gguf",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Deterministic token-hash → L2-normalized embedding (server-safe fallback).
pub fn hash_project_embedding(text: &str, d_model: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; d_model.max(1)];
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                tokens.push(std::mem::take(&mut cur));
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_string());
            }
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    if tokens.is_empty() {
        tokens.push("<empty>".into());
    }
    for tok in &tokens {
        let digest = Sha256::digest(tok.as_bytes());
        let idx = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize
            % vec.len();
        let sign = if digest[4] & 1 == 0 { 1.0 } else { -1.0 };
        let weight = 1.0 + (digest[5] as f32 / 255.0);
        vec[idx] += sign * weight;
    }
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let gram = &bytes[i..i + 3];
        let digest = Md5::digest(gram);
        let idx = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize
            % vec.len();
        vec[idx] += 0.15 * if digest[4] & 1 == 0 { 1.0 } else { -1.0 };
        i += 3;
    }
    l2_normalize(&mut vec);
    vec
}

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-8 {
        if let Some(x) = v.first_mut() {
            *x = 1.0;
        }
        return;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
}

pub fn format_qwen35_chat(messages: &[(&str, &str)], add_generation_prompt: bool, enable_thinking: bool) -> String {
    let mut parts = String::new();
    for (role, content) in messages {
        parts.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    if add_generation_prompt {
        parts.push_str("<|im_start|>assistant\n");
        if !enable_thinking {
            parts.push_str("<think>\n\n</think>\n");
        }
    }
    parts
}

pub struct GgufBackend {
    cfg: GgufConfig,
}

impl GgufBackend {
    pub fn new(cfg: GgufConfig) -> Self {
        Self { cfg }
    }

    pub fn from_env() -> Self {
        Self::new(GgufConfig::default())
    }

    /// Returns `(prompt, forced_prefix)` — prefix must be prepended to the completion.
    fn maybe_force_prefix(&self, prompt: &str) -> (String, &'static str) {
        if !self.cfg.force_search_prefix {
            return (prompt.to_string(), "");
        }
        let p = prompt;
        let tail = {
            let t = p.trim_end();
            let start = t.len().saturating_sub(80);
            &t[start..]
        };
        if tail.contains("<<<SEARCH") || tail.contains("<<<<<<< SEARCH") {
            return (p.to_string(), "<<<SEARCH\n");
        }
        (
            format!("{}\n<<<SEARCH\n", p.trim_end()),
            "<<<SEARCH\n",
        )
    }

    fn generate_via_server(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let url = format!(
            "{}/completion",
            self.cfg.server_url.trim_end_matches('/')
        );
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        });
        let agent = ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(180))
            .timeout_write(Duration::from_secs(30))
            .build();
        let resp = agent
            .post(&url)
            .set("Content-Type", "application/json")
            .send_json(body)
            .context("llama-server /completion")?;
        let v: serde_json::Value = resp.into_json().context("decode completion json")?;
        let content = v
            .get("content")
            .and_then(|c| c.as_str())
            .or_else(|| v.get("content").and_then(|c| c.as_array()).and_then(|a| a.first()).and_then(|x| x.as_str()))
            .unwrap_or("")
            .to_string();
        Ok(clean_cli_completion(&content))
    }

    fn generate_via_cli(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let bin = self
            .cfg
            .cli_bin
            .clone()
            .or_else(find_llama_cli)
            .ok_or_else(|| anyhow!("llama-cli not found"))?;
        let model = self
            .cfg
            .model_path
            .as_ref()
            .ok_or_else(|| anyhow!("no GGUF model path"))?;
        let out = Command::new(bin)
            .arg("-m")
            .arg(model)
            .arg("-p")
            .arg(prompt)
            .arg("-n")
            .arg(max_tokens.to_string())
            .arg("--temp")
            .arg("0")
            .arg("-ngl")
            .arg(self.cfg.n_gpu_layers.to_string())
            .arg("--no-display-prompt")
            .output()
            .context("llama-cli spawn")?;
        if !out.status.success() {
            return Err(anyhow!(
                "llama-cli failed: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(clean_cli_completion(&String::from_utf8_lossy(&out.stdout)))
    }

    fn server_alive(&self) -> bool {
        let url = format!("{}/health", self.cfg.server_url.trim_end_matches('/'));
        ureq::get(&url).call().is_ok()
    }
}

fn find_llama_cli() -> Option<PathBuf> {
    let cands = [
        "bin/prism-llama/llama-prism-b10777-e3de715/llama-cli",
        "../bin/prism-llama/llama-prism-b10777-e3de715/llama-cli",
    ];
    for c in cands {
        let p = PathBuf::from(c);
        if p.is_file() {
            return Some(p);
        }
    }
    which("llama-cli")
}

fn which(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        for p in env::split_paths(&paths) {
            let cand = p.join(name);
            if cand.is_file() {
                return Some(cand);
            }
        }
        None
    })
}

fn clean_cli_completion(text: &str) -> String {
    let mut t = text.to_string();
    // strip residual think blocks
    if let Some(idx) = t.find("</think>") {
        t = t[idx + "</think>".len()..].to_string();
    }
    t.trim().to_string()
}

impl CoreBackend for GgufBackend {
    fn encode_states(&self, text: &str) -> Result<Vec<f32>> {
        // Prefer deterministic hash projection — large AST batches crash some servers.
        Ok(hash_project_embedding(text, self.cfg.d_model))
    }

    fn generate_patch_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let (p, forced) = self.maybe_force_prefix(prompt);
        let join = |text: String| -> String {
            if forced.is_empty() {
                text
            } else if text.starts_with("<<<SEARCH") || text.starts_with("<<<<<<< SEARCH") {
                text
            } else {
                format!("{forced}{text}")
            }
        };
        if self.server_alive() {
            if let Ok(text) = self.generate_via_server(&p, max_tokens) {
                if !text.trim().is_empty() {
                    return Ok(join(text));
                }
            }
        }
        if self.cfg.model_path.as_ref().is_some_and(|p| Path::new(p).is_file()) {
            if let Ok(text) = self.generate_via_cli(&p, max_tokens) {
                return Ok(join(text));
            }
        }
        Err(anyhow!(
            "GGUF generate failed (server={} model={:?})",
            self.cfg.server_url,
            self.cfg.model_path
        ))
    }

    fn d_model(&self) -> usize {
        self.cfg.d_model
    }

    fn backend_kind(&self) -> &str {
        "gguf"
    }
}
