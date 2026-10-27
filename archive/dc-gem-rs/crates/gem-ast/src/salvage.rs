//! Multi-stage patch salvager (Phase-16.1 port).

use regex::Regex;
use thiserror::Error;

use crate::fuzzy::{
    apply_fuzzy_inplace, apply_line_substr_replace, apply_search_replace_fuzzy, leading_ws,
    unified_diff,
};

#[derive(Debug, Error)]
pub enum PatchError {
    #[error("all salvage stages failed")]
    AllFailed,
    #[error("empty source")]
    EmptySource,
}

#[derive(Clone, Debug, Default)]
pub struct SalvageMeta {
    pub stages_tried: Vec<u8>,
    pub stage: Option<u8>,
    pub error: Option<String>,
}

pub fn normalize_broken_diff_header(text: &str) -> String {
    let mut t = text.to_string();
    for (a, b) in [
        ("diff --git a/", "diff --git a/"),
        ("diff --gita/", "diff --git a/"),
        ("diff --gitb/", "diff --git b/"),
    ] {
        t = t.replace(a, b);
    }
    // also collapse "diff --git  a/" weird spacing already handled
    t = Regex::new(r"diff --git\s*a/")
        .unwrap()
        .replace_all(&t, "diff --git a/")
        .into_owned();
    t
}

/// `(path, old, new)` edits from loose SEARCH/REPLACE variants.
pub fn parse_loose_search_replace(text: &str) -> Vec<(String, String, String)> {
    let mut blocks: Vec<(String, String, String)> = Vec::new();

    // 1a) standard <<<<<<< SEARCH (no look-around — Rust regex limitation)
    let pat = Regex::new(
        r"(?is)(?:<<<\s*([^\n>]+)\s*>>>\s*)?<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
    )
    .unwrap();
    for m in pat.captures_iter(text) {
        let path = m.get(1).map(|x| x.as_str().trim().to_string()).unwrap_or_default();
        blocks.push((path, m[2].to_string(), m[3].to_string()));
    }

    // 1a2) GGUF forced prefix: <<<SEARCH\n ... \n=======\n ... \n>>>>>>> REPLACE
    // (also <<<path.py>>>\n<<<SEARCH\n ...)
    let pat_forced = Regex::new(
        r"(?is)(?:<<<\s*([^\n>]+?\.py)\s*>>>\s*\n)?<<<SEARCH\s*\n(.*?)\n=======\s*\n(.*?)(?:\n>>>>>>>\s*REPLACE|\z)",
    )
    .unwrap();
    for m in pat_forced.captures_iter(text) {
        let path = m.get(1).map(|x| x.as_str().trim().to_string()).unwrap_or_default();
        let old = m[2].trim_end_matches('\n').to_string();
        let new = m[3].trim_end_matches('\n').to_string();
        if !old.trim().is_empty() && !new.trim().is_empty() && old.trim() != new.trim() {
            blocks.push((path, old, new));
        }
    }

    // 1b) short SEARCH: / REPLACE: (consume until next marker or end)
    parse_labeled_blocks(
        text,
        &["SEARCH", "Search"],
        &["REPLACE", "Replace", "Updated", "=======", "---"],
        &mut blocks,
    );

    // 1c) Original: / Updated:
    parse_labeled_blocks(
        text,
        &["Original", "Before", "Old"],
        &["Updated", "After", "New", "Fixed"],
        &mut blocks,
    );

    // 1d) paired markdown fences
    let fence_re = Regex::new(r"(?is)```(?:python|py|diff|rust|rs)?\s*\n(.*?)```").unwrap();
    let fences: Vec<String> = fence_re
        .captures_iter(text)
        .map(|c| c[1].trim_end_matches('\n').to_string())
        .collect();
    if fences.len() >= 2 {
        let mut i = 0;
        while i + 1 < fences.len() {
            let a = &fences[i];
            let b = &fences[i + 1];
            if !a.trim().is_empty() && !b.trim().is_empty() && a.trim() != b.trim() {
                let a_l = a.trim_start();
                // single-def blocks → leave for stage-3 AST
                let is_single_def = a_l.starts_with("def ") && !a_l[4..].contains("def ");
                if !is_single_def {
                    blocks.push((String::new(), a.clone(), b.clone()));
                }
            }
            i += 2;
        }
    }

    // dedupe
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for b in blocks {
        let key = (b.0.clone(), b.1.trim().to_string(), b.2.trim().to_string());
        if key.1.is_empty() || !seen.insert(key.clone()) {
            continue;
        }
        out.push(b);
    }
    out
}

fn parse_labeled_blocks(
    text: &str,
    old_labels: &[&str],
    new_labels: &[&str],
    blocks: &mut Vec<(String, String, String)>,
) {
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        let is_old = old_labels.iter().any(|l| {
            trimmed.eq_ignore_ascii_case(l)
                || trimmed.eq_ignore_ascii_case(&format!("{l}:"))
                || trimmed.to_ascii_lowercase().starts_with(&format!("{}:", l.to_ascii_lowercase()))
        });
        if !is_old {
            i += 1;
            continue;
        }
        i += 1;
        let mut old_lines = Vec::new();
        while i < lines.len() {
            let t = lines[i].trim();
            let is_new = new_labels.iter().any(|l| {
                t.eq_ignore_ascii_case(l)
                    || t.eq_ignore_ascii_case(&format!("{l}:"))
                    || t.starts_with("===")
                    || t.starts_with("---")
                    || t.to_ascii_lowercase().starts_with(&format!("{}:", l.to_ascii_lowercase()))
            });
            if is_new {
                break;
            }
            old_lines.push(lines[i]);
            i += 1;
        }
        if i >= lines.len() {
            break;
        }
        i += 1; // skip new label
        let mut new_lines = Vec::new();
        while i < lines.len() {
            let t = lines[i].trim();
            if old_labels.iter().any(|l| {
                t.eq_ignore_ascii_case(l)
                    || t.eq_ignore_ascii_case(&format!("{l}:"))
                    || t.starts_with("###")
                    || t.starts_with("<<<<<<<")
            }) {
                break;
            }
            new_lines.push(lines[i]);
            i += 1;
        }
        let old = old_lines.join("\n").trim_end_matches('\n').to_string();
        let new = new_lines.join("\n").trim_end_matches('\n').to_string();
        if !old.trim().is_empty() && !new.trim().is_empty() && old.trim() != new.trim() {
            blocks.push((String::new(), old, new));
        }
    }
}

/// Extract `(old, new)` from `+/-` unified / naked diff lines.
pub fn extract_plus_minus_edits(text: &str) -> Vec<(String, String)> {
    let t = normalize_broken_diff_header(text);
    let mut edits = Vec::new();
    let mut old_buf: Vec<String> = Vec::new();
    let mut new_buf: Vec<String> = Vec::new();

    let flush = |old_buf: &mut Vec<String>, new_buf: &mut Vec<String>, edits: &mut Vec<(String, String)>| {
        if old_buf.is_empty() && new_buf.is_empty() {
            return;
        }
        let o = old_buf.join("\n");
        let n = new_buf.join("\n");
        if !o.trim().is_empty() && !n.trim().is_empty() && o.trim() != n.trim() {
            edits.push((o, n));
        }
        old_buf.clear();
        new_buf.clear();
    };

    for ln in t.lines() {
        if ln.starts_with("---") || ln.starts_with("+++") || ln.starts_with("@@") || ln.starts_with("diff ")
        {
            flush(&mut old_buf, &mut new_buf, &mut edits);
            continue;
        }
        if ln.starts_with('-') && !ln.starts_with("---") {
            old_buf.push(ln[1..].to_string());
            continue;
        }
        if ln.starts_with('+') && !ln.starts_with("+++") {
            new_buf.push(ln[1..].to_string());
            continue;
        }
        if !old_buf.is_empty() || !new_buf.is_empty() {
            flush(&mut old_buf, &mut new_buf, &mut edits);
        }
    }
    flush(&mut old_buf, &mut new_buf, &mut edits);
    edits
}

fn first_function_name(source: &str) -> Option<String> {
    let re = Regex::new(r"(?m)^\s*(?:async\s+)?(?:def|fn)\s+(\w+)").unwrap();
    re.captures(source).map(|c| c[1].to_string())
}

fn extract_method_sources(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let fence_re = Regex::new(r"(?is)```(?:python|py|rust|rs)?\s*\n(.*?)```").unwrap();
    for m in fence_re.captures_iter(text) {
        let body = m[1].trim_matches('\n').to_string();
        if Regex::new(r"(?m)^\s*(?:async\s+)?(?:def|fn)\s+\w+")
            .unwrap()
            .is_match(&body)
        {
            chunks.push(body);
        }
    }
    if !chunks.is_empty() {
        return chunks;
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    let def_re = Regex::new(r"^\s*(?:async\s+)?(?:def|fn)\s+\w+").unwrap();
    while i < lines.len() {
        if def_re.is_match(lines[i]) {
            let base = leading_ws(lines[i]);
            let mut block = vec![lines[i].to_string()];
            i += 1;
            while i < lines.len() {
                let ln = lines[i];
                if ln.trim().is_empty() {
                    block.push(ln.to_string());
                    i += 1;
                    continue;
                }
                let ws = leading_ws(ln);
                if ws.len() <= base.len()
                    && !ln.trim().starts_with('#')
                    && !ln.trim().starts_with('@')
                {
                    if Regex::new(r"^\s*(?:async\s+)?(?:def|fn|class|struct|impl|trait)\s+")
                        .unwrap()
                        .is_match(ln)
                    {
                        break;
                    }
                    if !ln.starts_with(&base) && !ln.trim().is_empty() {
                        break;
                    }
                }
                block.push(ln.to_string());
                i += 1;
            }
            let mut src = block.join("\n");
            if !src.ends_with('\n') {
                src.push('\n');
            }
            if src.contains("def ") || src.contains("fn ") {
                chunks.push(src);
            }
            continue;
        }
        i += 1;
    }
    chunks
}

/// Stage-3: replace function/method by name, preserving base indent.
pub fn replace_ast_symbol_in_file(
    file_text: &str,
    new_def_source: &str,
    path: &str,
    symbol_hint: Option<&str>,
) -> Option<String> {
    let updated = replace_ast_inplace(file_text, new_def_source, symbol_hint)?;
    unified_diff(file_text, &updated, path)
}

pub fn replace_ast_inplace(
    file_text: &str,
    new_def_source: &str,
    symbol_hint: Option<&str>,
) -> Option<String> {
    let mut name = symbol_hint
        .map(|s| s.to_string())
        .or_else(|| first_function_name(new_def_source))?;
    if let Some(leaf) = name.rsplit('.').next() {
        name = leaf.to_string();
    }
    let lines: Vec<&str> = file_text.split_inclusive('\n').collect();
    let no_nl: Vec<String> = lines
        .iter()
        .map(|l| l.trim_end_matches('\n').to_string())
        .collect();

    // Find def/fn line
    let def_pat = Regex::new(&format!(
        r"^(\s*)(?:async\s+)?(?:def|fn)\s+{}\b",
        regex::escape(&name)
    ))
    .ok()?;
    let mut start = None;
    for (i, ln) in no_nl.iter().enumerate() {
        if def_pat.is_match(ln) {
            start = Some(i);
            break;
        }
    }
    let mut start = start?;
    // include decorators
    while start > 0 {
        let prev = no_nl[start - 1].trim();
        if prev.starts_with('@') || prev.is_empty() {
            start -= 1;
            if prev.starts_with('@') {
                continue;
            }
            // blank between decorators — keep scanning one more?
            if start > 0 && no_nl[start - 1].trim().starts_with('@') {
                continue;
            }
            start += 1; // don't eat unrelated blank
            break;
        }
        break;
    }

    let def_line = &no_nl[start];
    let base_indent = leading_ws(def_line);
    // find end: next line with indent <= base that starts a top-level construct
    let mut end = start + 1;
    while end < no_nl.len() {
        let ln = &no_nl[end];
        if ln.trim().is_empty() {
            end += 1;
            continue;
        }
        let ws = leading_ws(ln);
        if ws.len() <= base_indent.len() {
            break;
        }
        end += 1;
    }
    // trim trailing blanks from range
    while end > start + 1 && no_nl[end - 1].trim().is_empty() {
        end -= 1;
    }

    let new_src = format!("{}\n", new_def_source.trim_matches('\n'));
    let new_lines: Vec<&str> = new_src.lines().collect();
    if new_lines.is_empty() {
        return None;
    }
    let first_ws = leading_ws(new_lines[0]);
    let mut realigned = Vec::new();
    for ln in &new_lines {
        if ln.trim().is_empty() {
            realigned.push(String::new());
            continue;
        }
        let cur = leading_ws(ln);
        let rel = if !first_ws.is_empty() && cur.starts_with(&first_ws) {
            &cur[first_ws.len()..]
        } else {
            cur.as_str()
        };
        realigned.push(format!("{base_indent}{rel}{}", ln.trim()));
    }

    let mut out = String::new();
    for line in &lines[..start] {
        out.push_str(line);
    }
    for ln in &realigned {
        out.push_str(ln);
        out.push('\n');
    }
    for line in &lines[end..] {
        out.push_str(line);
    }
    if out == file_text {
        return None;
    }
    Some(out)
}

fn build_fuzzy_from_edits(
    file_text: &str,
    path: &str,
    edits: &[(String, String, String)],
) -> Option<String> {
    let mut cur = file_text.to_string();
    let mut any = false;
    for (epath, old, new) in edits {
        if !epath.is_empty() {
            let base = std::path::Path::new(epath)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let path_base = std::path::Path::new(path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if epath != path && base != path_base && base != path {
                // try anyway if only one file
            }
        }
        if let Some(trial) = apply_fuzzy_inplace(&cur, old, new, 0.72) {
            cur = trial;
            any = true;
        }
    }
    if any && cur != file_text {
        unified_diff(file_text, &cur, path)
    } else {
        None
    }
}

/// 4-stage salvage → unified diff (`git apply` compatible).
///
/// Stage 4 (GGUF `<<<SEARCH\n` prefix) lives in `gem-engine`.
pub fn salvage_and_apply_patch(
    file_path: &str,
    original_source: &str,
    raw_model_output: &str,
) -> Result<(String, SalvageMeta), PatchError> {
    if original_source.is_empty() {
        return Err(PatchError::EmptySource);
    }
    let mut meta = SalvageMeta::default();
    let raw = normalize_broken_diff_header(raw_model_output);

    // Stage 1: loose SEARCH/REPLACE
    meta.stages_tried.push(1);
    let edits = parse_loose_search_replace(&raw);
    if !edits.is_empty() {
        if let Some(patch) = build_fuzzy_from_edits(original_source, file_path, &edits) {
            meta.stage = Some(1);
            return Ok((patch, meta));
        }
    }

    // Stage 2: +/- edits
    meta.stages_tried.push(2);
    let pm = extract_plus_minus_edits(&raw);
    if !pm.is_empty() {
        let pm_edits: Vec<(String, String, String)> = pm
            .iter()
            .map(|(o, n)| (String::new(), o.clone(), n.clone()))
            .collect();
        if let Some(patch) = build_fuzzy_from_edits(original_source, file_path, &pm_edits) {
            meta.stage = Some(2);
            return Ok((patch, meta));
        }
        let mut cur = original_source.to_string();
        let mut hit_any = false;
        for (o, n) in &pm {
            if let Some(d) = apply_search_replace_fuzzy(&cur, o, n, file_path, 0.55) {
                let _ = d;
                if let Some(trial) = apply_fuzzy_inplace(&cur, o, n, 0.55) {
                    cur = trial;
                    hit_any = true;
                    continue;
                }
            }
            if let Some(trial2) = apply_line_substr_replace(&cur, o, n) {
                if trial2 != cur {
                    cur = trial2;
                    hit_any = true;
                }
            }
        }
        if hit_any {
            if let Some(patch) = unified_diff(original_source, &cur, file_path) {
                meta.stage = Some(2);
                return Ok((patch, meta));
            }
        }
    }

    // Stage 3: AST / symbol whole-function replace
    meta.stages_tried.push(3);
    let methods = extract_method_sources(&raw);
    if !methods.is_empty() {
        let mut cur = original_source.to_string();
        let mut applied = false;
        for meth in &methods {
            let fname = first_function_name(meth);
            let hints: Vec<String> = fname.into_iter().collect();
            for hint in &hints {
                if !(cur.contains(&format!("def {hint}"))
                    || cur.contains(&format!("async def {hint}"))
                    || cur.contains(&format!("fn {hint}")))
                {
                    continue;
                }
                if let Some(trial) = replace_ast_inplace(&cur, meth, Some(hint)) {
                    cur = trial;
                    applied = true;
                    break;
                }
            }
            if applied {
                break;
            }
        }
        if applied {
            if let Some(patch) = unified_diff(original_source, &cur, file_path) {
                meta.stage = Some(3);
                return Ok((patch, meta));
            }
        }
    }

    meta.error = Some("all stages failed".into());
    Err(PatchError::AllFailed)
}
