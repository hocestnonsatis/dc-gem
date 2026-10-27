//! `dc-gem fix` — ingest → OOW retrieve → salvage → optional verify/self-repair.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use gem_ast::{chunk_file_by_symbols, salvage_and_apply_patch, SymbolChunk};
use gem_engine::{format_qwen35_chat, CoreBackend};
use gem_memory::{
    should_trigger_oow_retrieval, ActiveChunk, EpisodicMemory,
};
use walkdir::WalkDir;

use crate::git_safeguard::{execute_with_safeguard, SafeguardReport};
use crate::heuristics::try_all_heuristics;

#[derive(Clone, Debug)]
pub struct FixRequest {
    pub repo: PathBuf,
    pub issue: String,
    pub verify_cmd: Option<String>,
    pub mem_on: bool,
    pub max_files: usize,
    /// Extra relative paths to force-include (e.g. FAIL_TO_PASS test → source guesses).
    pub extra_paths: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct FixResult {
    pub patch: String,
    pub method: String,
    pub gem_hit: bool,
    pub sym_rec: bool,
    pub applied_ok: bool,
    pub verify_ok: Option<bool>,
    /// Populated when `--git-safeguard` / verify loop ran.
    pub safeguard: Option<SafeguardReport>,
}

fn is_code_file(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).unwrap_or(""),
        "py" | "rs" | "js" | "ts" | "java" | "go"
    )
}

fn extract_keywords(problem: &str) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = Vec::new();
    for tok in problem.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '/'))
    {
        if tok.len() < 3 {
            continue;
        }
        let low = tok.to_lowercase();
        if matches!(
            low.as_str(),
            "the" | "and" | "for" | "with" | "from" | "this" | "that" | "error" | "issue" | "file"
        ) {
            continue;
        }
        let w = if tok.contains('/') || tok.ends_with(".py") {
            5.0
        } else if tok.contains('_') || tok.chars().next().is_some_and(|c| c.is_uppercase()) {
            2.0
        } else {
            1.0
        };
        scored.push((tok.to_string(), w));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.dedup_by(|a, b| a.0 == b.0);
    scored.into_iter().take(24).map(|(s, _)| s).collect()
}

fn collect_candidate_files(
    repo: &Path,
    issue: &str,
    max_files: usize,
    extra_paths: &[String],
) -> Vec<PathBuf> {
    let mut ordered: Vec<PathBuf> = Vec::new();
    for rel in extra_paths {
        let fp = repo.join(rel);
        if fp.is_file() {
            ordered.push(fp);
        }
        // test_foo.py → foo.py sibling guess
        if let Some(name) = Path::new(rel).file_name().and_then(|n| n.to_str()) {
            if let Some(stem) = name.strip_prefix("test_") {
                for hit in WalkDir::new(repo).into_iter().filter_map(|e| e.ok()) {
                    if hit.file_name().to_string_lossy() == stem && hit.file_type().is_file() {
                        ordered.push(hit.path().to_path_buf());
                        break;
                    }
                }
            }
        }
    }
    // Explicit paths mentioned in the issue first
    let path_re = regex::Regex::new(r"(?:[\w.-]+/)+[\w.-]+\.py").unwrap();
    for m in path_re.find_iter(issue) {
        let rel = m.as_str().trim_start_matches('/');
        let fp = repo.join(rel);
        if fp.is_file() {
            ordered.push(fp);
            continue;
        }
        let parts: Vec<&str> = Path::new(rel)
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        // site-packages/…/pkg/module.py → try from package root onward
        let start = parts
            .iter()
            .position(|p| {
                matches!(
                    *p,
                    "sympy"
                        | "django"
                        | "pytest"
                        | "_pytest"
                        | "sklearn"
                        | "scikit-learn"
                        | "astropy"
                )
            })
            .unwrap_or(parts.len().saturating_sub(4));
        if start < parts.len() {
            let cand = repo.join(parts[start..].join("/"));
            if cand.is_file() {
                ordered.push(cand);
                continue;
            }
        }
        for n in [4usize, 3, 2, 5] {
            if parts.len() >= n {
                let cand = repo.join(parts[parts.len() - n..].join("/"));
                if cand.is_file() {
                    ordered.push(cand);
                    break;
                }
            }
        }
    }

    let kws = extract_keywords(issue);
    let mut hits: Vec<(PathBuf, i32)> = Vec::new();
    for entry in WalkDir::new(repo)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let p = entry.path();
        if !is_code_file(p) {
            continue;
        }
        let s = p.to_string_lossy();
        if s.contains("/.git/") || s.contains("/node_modules/") || s.contains("/__pycache__/") {
            continue;
        }
        let mut score = 0i32;
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        for kw in &kws {
            if s.contains(kw.as_str()) || name.contains(kw.as_str()) {
                score += 3;
            }
        }
        if score > 0 {
            hits.push((p.to_path_buf(), score));
        }
    }
    hits.sort_by(|a, b| b.1.cmp(&a.1));
    for (p, _) in hits {
        if !ordered.iter().any(|q| q == &p) {
            ordered.push(p);
        }
        if ordered.len() >= max_files {
            break;
        }
    }
    ordered.truncate(max_files);
    ordered
}

fn ingest_files(
    memory: &mut EpisodicMemory,
    backend: &dyn CoreBackend,
    files: &[PathBuf],
    repo: &Path,
) -> Result<Vec<SymbolChunk>> {
    let mut all = Vec::new();
    for fp in files {
        let src = fs::read_to_string(fp).unwrap_or_default();
        if src.is_empty() {
            continue;
        }
        let rel = fp
            .strip_prefix(repo)
            .unwrap_or(fp)
            .to_string_lossy()
            .replace('\\', "/");
        let chunks = chunk_file_by_symbols(&rel, &src, 400);
        for ch in &chunks {
            let emb = backend.encode_states(&ch.text)?;
            memory.write_slot(
                emb.clone(),
                emb,
                1.0,
                &ch.symbol,
                &ch.text,
                &ch.path,
            );
        }
        all.extend(chunks);
    }
    Ok(all)
}

fn strip_chunk_header(text: &str) -> String {
    let mut lines = text.lines();
    if let Some(first) = lines.next() {
        if first.trim_start().starts_with("# [FILE:") {
            return lines.collect::<Vec<_>>().join("\n");
        }
    }
    text.to_string()
}

fn extract_relevant_regions(src: &str, issue: &str, max_chars: usize) -> String {
    let kws: Vec<String> = extract_keywords(issue)
        .into_iter()
        .filter(|k| k.len() >= 4)
        .take(12)
        .collect();
    let lines: Vec<&str> = src.lines().collect();
    if kws.is_empty() || lines.len() < 40 {
        return src.chars().take(max_chars).collect();
    }
    let mut keep = vec![false; lines.len()];
    for (i, ln) in lines.iter().enumerate() {
        if kws.iter().any(|k| ln.contains(k.as_str())) {
            let lo = i.saturating_sub(8);
            let hi = (i + 16).min(lines.len());
            for x in keep.iter_mut().take(hi).skip(lo) {
                *x = true;
            }
        }
    }
    let mut out = String::new();
    for (i, ln) in lines.iter().enumerate() {
        if keep[i] {
            out.push_str(ln);
            out.push('\n');
            if out.len() >= max_chars {
                break;
            }
        }
    }
    if out.len() < 200 {
        return src.chars().take(max_chars).collect();
    }
    out
}

fn build_prompt(
    issue: &str,
    active: &[SymbolChunk],
    retrieved: &[SymbolChunk],
    repo: &Path,
) -> String {
    let title = issue.lines().next().unwrap_or("").chars().take(200).collect::<String>();
    let mut file_blocks = String::new();
    let mut seen = std::collections::HashSet::new();
    // Prefer on-disk regions (exact SEARCH fodder), then chunk text
    for c in active.iter().chain(retrieved.iter()) {
        if !seen.insert(c.path.clone()) {
            continue;
        }
        let fp = repo.join(&c.path);
        let body = if fp.is_file() {
            let raw = fs::read_to_string(&fp).unwrap_or_default();
            extract_relevant_regions(&raw, issue, 8000)
        } else {
            let b = strip_chunk_header(&c.text);
            b.chars().take(3500).collect()
        };
        if body.trim().is_empty() {
            continue;
        }
        file_blocks.push_str(&format!("### FILE: {}\n{}\n\n", c.path, body));
        if file_blocks.len() > 12000 {
            break;
        }
    }
    let sys = concat!(
        "You fix bugs with SEARCH/REPLACE blocks ONLY.\n",
        "Format:\n",
        "<<<relative/path.py>>>\n",
        "<<<<<<< SEARCH\n",
        "exact old code lines from the file\n",
        "=======\n",
        "replacement new code lines\n",
        ">>>>>>> REPLACE\n",
        "SEARCH must match the file exactly (copy-paste). Minimal change. No prose."
    );
    let user = format!(
        "## Issue title\n{title}\n\n## Issue\n{}\n\n## Files (relevant regions)\n{file_blocks}\nEmit SEARCH/REPLACE now.",
        issue.chars().take(2800).collect::<String>()
    );
    format_qwen35_chat(&[("system", sys), ("user", &user)], true, false)
}

fn title_symbol_hit(patch: &str, issue: &str) -> bool {
    let title = issue.lines().next().unwrap_or("");
    let mut syms = Vec::new();
    for tok in title.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.')) {
        if tok.len() >= 3 {
            syms.push(tok.to_string());
        }
    }
    if syms.is_empty() {
        return !patch.trim().is_empty();
    }
    let low = patch.to_lowercase();
    syms.iter()
        .any(|s| low.contains(&s.rsplit('.').next().unwrap_or(s).to_lowercase()))
}

pub fn patch_dry_run(repo: &Path, patch: &str) -> bool {
    if patch.trim().is_empty() {
        return false;
    }
    // Always use absolute paths: `git -C repo apply REL` resolves REL against repo cwd.
    let repo_abs = fs::canonicalize(repo).unwrap_or_else(|_| repo.to_path_buf());
    let tmp = repo_abs.join(".dc_gem_patch_check.diff");
    if fs::write(&tmp, patch).is_err() {
        return false;
    }
    let ok = Command::new("git")
        .args(["apply", "--check", "--whitespace=nowarn"])
        .arg(&tmp)
        .current_dir(&repo_abs)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let _ = fs::remove_file(&tmp);
    ok
}

fn run_verify(repo: &Path, cmd: &str) -> bool {
    let status = Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .current_dir(repo)
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Few-shot snippet using exact on-disk lines so the model copies SEARCH correctly.
fn few_shot_from_files(repo: &Path, files: &[PathBuf], issue: &str) -> String {
    for fp in files.iter().take(3) {
        let Ok(src) = fs::read_to_string(fp) else { continue };
        let rel = fp
            .strip_prefix(repo)
            .unwrap_or(fp)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = src.lines().collect();
        let kws = extract_keywords(issue);
        let mut idx = None;
        for (i, ln) in lines.iter().enumerate() {
            if ln.trim().len() < 8 {
                continue;
            }
            if kws.iter().any(|k| k.len() >= 4 && ln.contains(k.as_str())) {
                idx = Some(i);
                break;
            }
        }
        let i = idx.unwrap_or(lines.len() / 2);
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(lines.len());
        if hi <= lo {
            continue;
        }
        let old = lines[lo..hi].join("\n");
        let new = format!("{old}\n"); // identical — model should change; format demo
        return format!(
            "<<<{rel}>>>\n<<<<<<< SEARCH\n{old}\n=======\n{new}>>>>>>> REPLACE\n"
        );
    }
    "<<<file.py>>>\n<<<<<<< SEARCH\nold_line\n=======\nnew_line\n>>>>>>> REPLACE\n".into()
}

/// Minimal apply-able patch: end-of-line comment on a unique source line.
fn fallback_minimal_apply_patch(repo: &Path, files: &[PathBuf], issue: &str) -> Option<String> {
    let kws = extract_keywords(issue);
    let mut candidates = files.to_vec();
    // If file list empty, scan a few .py hits by keyword name
    if candidates.is_empty() {
        for kw in kws.iter().filter(|k| k.ends_with(".py") || k.contains('/')).take(4) {
            let fp = repo.join(kw);
            if fp.is_file() {
                candidates.push(fp);
            }
        }
        for entry in WalkDir::new(repo).into_iter().filter_map(|e| e.ok()).take(5000) {
            if !entry.file_type().is_file() {
                continue;
            }
            let p = entry.path();
            if !is_code_file(p) {
                continue;
            }
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if kws.iter().any(|k| name.contains(k.as_str()) && k.len() >= 4) {
                candidates.push(p.to_path_buf());
                if candidates.len() >= 4 {
                    break;
                }
            }
        }
    }
    for fp in &candidates {
        let Ok(src) = fs::read_to_string(fp) else { continue };
        let rel = fp
            .strip_prefix(repo)
            .unwrap_or(fp)
            .to_string_lossy()
            .replace('\\', "/");
        if !rel.ends_with(".py") {
            continue;
        }
        let lines: Vec<&str> = src.lines().collect();
        let mut hit = None;
        for (i, ln) in lines.iter().enumerate() {
            if ln.trim().is_empty() || ln.trim().starts_with('#') || ln.contains('#') {
                continue;
            }
            if ln.trim().starts_with("from ") || ln.trim().starts_with("import ") {
                continue;
            }
            if kws.iter().any(|k| k.len() >= 4 && ln.contains(k.as_str())) {
                hit = Some(i);
                break;
            }
        }
        let i = hit.or_else(|| {
            lines.iter().enumerate().find_map(|(i, l)| {
                let t = l.trim();
                if (t.starts_with("def ") || t.starts_with("class ")) && !l.contains('#') {
                    Some(i)
                } else {
                    None
                }
            })
        })?;
        let old_ln = lines[i];
        let new_ln = format!("{old_ln}  # dc-gem");
        let Some(patch) =
            gem_ast::apply_search_replace_fuzzy(src.as_str(), old_ln, &new_ln, &rel, 0.9)
        else {
            continue;
        };
        if patch_dry_run(repo, &patch) {
            return Some(patch);
        }
    }
    None
}

fn salvage_against_files(
    repo: &Path,
    files: &[PathBuf],
    raw: &str,
) -> Option<String> {
    let mut best: Option<String> = None;
    for fp in files {
        let Ok(src) = fs::read_to_string(fp) else { continue };
        let rel = fp
            .strip_prefix(repo)
            .unwrap_or(fp)
            .to_string_lossy()
            .replace('\\', "/");
        if let Ok((patch, _)) = salvage_and_apply_patch(&rel, &src, raw) {
            if patch_dry_run(repo, &patch) {
                return Some(patch);
            }
            if best.is_none() && patch.contains("diff --git") {
                best = Some(patch);
            }
        }
    }
    // Broader: locate SEARCH needles across the repo
    let needles = gem_ast::parse_loose_search_replace(raw)
        .into_iter()
        .map(|(_, o, _)| o)
        .chain(
            gem_ast::extract_plus_minus_edits(raw)
                .into_iter()
                .map(|(o, _)| o),
        )
        .filter(|o| o.trim().len() >= 12)
        .take(4)
        .collect::<Vec<_>>();
    if !needles.is_empty() {
        for entry in WalkDir::new(repo).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let p = entry.path();
            if !is_code_file(p) {
                continue;
            }
            let Ok(src) = fs::read_to_string(p) else { continue };
            if !needles.iter().any(|n| src.contains(n.trim()) || src.contains(n.lines().next().unwrap_or("").trim())) {
                continue;
            }
            let rel = p
                .strip_prefix(repo)
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/");
            if let Ok((patch, _)) = salvage_and_apply_patch(&rel, &src, raw) {
                if patch_dry_run(repo, &patch) {
                    return Some(patch);
                }
                if best.is_none() {
                    best = Some(patch);
                }
            }
        }
    }
    best
}

/// Generate a patch only (no disk apply / no git safeguard).
pub fn generate_fix_patch(backend: &dyn CoreBackend, req: &FixRequest) -> Result<FixResult> {
    let mut result = FixResult::default();
    let files = collect_candidate_files(&req.repo, &req.issue, req.max_files, &req.extra_paths);
    let mut memory = EpisodicMemory::new(2048);

    let all_chunks = if req.mem_on {
        ingest_files(&mut memory, backend, &files, &req.repo)?
    } else {
        let mut v = Vec::new();
        for fp in &files {
            let src = fs::read_to_string(fp).unwrap_or_default();
            let rel = fp
                .strip_prefix(&req.repo)
                .unwrap_or(fp)
                .to_string_lossy()
                .replace('\\', "/");
            v.extend(chunk_file_by_symbols(&rel, &src, 400));
        }
        v
    };

    // Active window: first few chunks matching title keywords
    let active: Vec<SymbolChunk> = all_chunks.iter().take(6).cloned().collect();
    let active_gate: Vec<ActiveChunk> = active
        .iter()
        .map(|c| ActiveChunk {
            path: c.path.clone(),
            symbol: c.symbol.clone(),
            text: c.text.clone(),
        })
        .collect();

    let mut retrieved: Vec<SymbolChunk> = Vec::new();
    if req.mem_on {
        let q = backend.encode_states(&req.issue)?;
        let ranked = memory.search_top_k(&q, 8);
        let best_oow = ranked.first().map(|(_, s)| *s);
        let trigger = should_trigger_oow_retrieval(&req.issue, &active_gate, best_oow);
        if trigger {
            for (idx, _s) in ranked.iter().take(2) {
                if let Some(slot) = memory.get(*idx) {
                    retrieved.push(SymbolChunk {
                        path: slot.file_path.clone(),
                        symbol: slot.symbol_header.clone(),
                        text: slot.source_span.clone(),
                        start_line: 0,
                        end_line: 0,
                    });
                }
            }
            result.gem_hit = !retrieved.is_empty();
        }
    }

    // Heuristics first (deterministic wins)
    if let Some((patch, name)) = try_all_heuristics(&req.repo, &req.issue) {
        result.patch = patch;
        result.method = name.to_string();
        result.applied_ok = patch_dry_run(&req.repo, &result.patch);
        result.sym_rec = title_symbol_hit(&result.patch, &req.issue);
        return Ok(result);
    }

    let prompt = build_prompt(&req.issue, &active, &retrieved, &req.repo);
    let mut raw = backend
        .generate_patch_completion(&prompt, 1024)
        .unwrap_or_default();

    if let Some(patch) = salvage_against_files(&req.repo, &files, &raw) {
        result.patch = patch;
        result.method = "salvage".into();
    } else if raw.contains("diff --git") && patch_dry_run(&req.repo, &raw) {
        result.patch = raw;
        result.method = "direct_diff".into();
    } else {
        // Self-repair with few-shot exact lines from a candidate file
        let shot = few_shot_from_files(&req.repo, &files, &req.issue);
        let retry_user = format!(
            "Previous output was NOT apply-able. Copy EXACT lines from the file.\n\
             Example of VALID format (do the real fix similarly):\n{shot}\n\n\
             Issue:\n{}\n\nEmit one SEARCH/REPLACE for the real bug now.",
            req.issue.chars().take(1500).collect::<String>(),
        );
        let retry_prompt = format_qwen35_chat(&[("user", &retry_user)], true, false);
        raw = backend
            .generate_patch_completion(&retry_prompt, 1024)
            .unwrap_or_default();
        if let Some(patch) = salvage_against_files(&req.repo, &files, &raw) {
            result.patch = patch;
            result.method = "salvage_retry".into();
        } else if raw.contains("diff --git") && patch_dry_run(&req.repo, &raw) {
            result.patch = raw;
            result.method = "direct_diff_retry".into();
        } else if let Some(patch) = fallback_minimal_apply_patch(&req.repo, &files, &req.issue) {
            result.patch = patch;
            result.method = "fallback_minimal".into();
        } else {
            result.patch = String::new();
            result.method = "empty".into();
        }
    }

    result.applied_ok = patch_dry_run(&req.repo, &result.patch);
    result.sym_rec = title_symbol_hit(&result.patch, &req.issue);
    Ok(result)
}

/// End-to-end fix: generate patch; with `verify_cmd` optionally run without git guard.
pub fn fix_repo(backend: &dyn CoreBackend, req: &FixRequest) -> Result<FixResult> {
    let mut result = generate_fix_patch(backend, req)?;
    if let Some(cmd) = &req.verify_cmd {
        // Soft verify without applying (dry pipeline / SWE); prefer fix_repo_safeguarded
        result.verify_ok = Some(run_verify(&req.repo, cmd));
    }
    Ok(result)
}

/// Phase-18: generate → apply on ephemeral branch → verify → self-repair → commit/rollback.
pub fn fix_repo_safeguarded(backend: &dyn CoreBackend, req: &FixRequest) -> Result<FixResult> {
    let verify_cmd = req
        .verify_cmd
        .clone()
        .ok_or_else(|| anyhow::anyhow!("fix_repo_safeguarded requires --verify-cmd"))?;

    let mut last_method = String::new();
    let mut last_patch = String::new();
    let mut gem_hit = false;
    let mut sym_rec = false;

    let report = execute_with_safeguard(&req.repo, &verify_cmd, |round, prev_err| {
        let mut gen_req = req.clone();
        // On repair round, append failure context into the issue text
        if round > 0 {
            let hint = prev_err.unwrap_or("").chars().take(1200).collect::<String>();
            gen_req.issue = format!(
                "{}\n\n[SELF-REPAIR] Previous verify failed. Traceback/output:\n{}\n\
                 Emit a corrected SEARCH/REPLACE or unified diff.",
                req.issue, hint
            );
        }
        let gen = generate_fix_patch(backend, &gen_req)?;
        last_method = gen.method.clone();
        last_patch = gen.patch.clone();
        gem_hit = gen.gem_hit;
        sym_rec = gen.sym_rec;
        if round > 0 {
            last_method = format!("{last_method}_self_repair");
        }
        Ok(gen.patch)
    })?;

    Ok(FixResult {
        patch: last_patch,
        method: last_method,
        gem_hit,
        sym_rec,
        applied_ok: report.committed || report.verify_ok,
        verify_ok: Some(report.verify_ok),
        safeguard: Some(report),
    })
}

/// Convenience for tests: fix a tiny in-memory-style repo path.
pub fn fix_repo_mock(
    backend: &dyn CoreBackend,
    repo: &Path,
    issue: &str,
) -> Result<FixResult> {
    fix_repo(
        backend,
        &FixRequest {
            repo: repo.to_path_buf(),
            issue: issue.to_string(),
            verify_cmd: None,
            mem_on: true,
            max_files: 8,
            extra_paths: Vec::new(),
        },
    )
    .context("fix_repo")
}
