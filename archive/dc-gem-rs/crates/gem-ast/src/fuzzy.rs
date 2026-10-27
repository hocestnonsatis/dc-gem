//! Indentation-tolerant SEARCH/REPLACE and unified diff helpers.

use similar::{ChangeTag, TextDiff};

pub fn leading_ws(s: &str) -> String {
    s.chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .collect()
}

pub fn unified_diff(original: &str, updated: &str, path: &str) -> Option<String> {
    if original == updated {
        return None;
    }
    let diff = TextDiff::from_lines(original, updated);
    let mut body = String::new();
    // similar's unified diff
    let ud = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    if ud.trim().is_empty() {
        return None;
    }
    body.push_str(&format!("diff --git a/{path} b/{path}\n"));
    body.push_str(&ud);
    // Ensure trailing newline
    if !body.ends_with('\n') {
        body.push('\n');
    }
    Some(body)
}

fn strip_lines(block: &str) -> Vec<String> {
    block.lines().map(|l| l.to_string()).collect()
}

fn reindent_replace(
    replace_lines: &[String],
    src_indents: &[String],
    base_indent: &str,
    search_base: &str,
) -> Vec<String> {
    let mut out = Vec::new();
    for (i, ln) in replace_lines.iter().enumerate() {
        if ln.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let src_ind = src_indents
            .get(i)
            .cloned()
            .unwrap_or_else(|| base_indent.to_string());
        let cur = leading_ws(ln);
        if !search_base.is_empty() && cur.starts_with(search_base) {
            let rel = &cur[search_base.len()..];
            out.push(format!("{base_indent}{rel}{}", ln.trim()));
        } else {
            out.push(format!("{src_ind}{}", ln.trim()));
        }
    }
    out
}

fn sequence_ratio(a: &[String], b: &[String]) -> f32 {
    // simple SequenceMatcher-like ratio via similar
    let left = a.join("\n");
    let right = b.join("\n");
    let diff = TextDiff::from_lines(&left, &right);
    let mut same = 0usize;
    let mut total = 0usize;
    for change in diff.iter_all_changes() {
        let n = change.value().lines().count().max(1);
        total += n;
        if change.tag() == ChangeTag::Equal {
            same += n;
        }
    }
    if total == 0 {
        return 1.0;
    }
    (2.0 * same as f32) / (a.len() + b.len()).max(1) as f32
}

/// Fuzzy SEARCH/REPLACE → unified diff. Returns `None` on failure.
pub fn apply_search_replace_fuzzy(
    file_text: &str,
    search: &str,
    replace: &str,
    path: &str,
    min_ratio: f32,
) -> Option<String> {
    let updated = apply_fuzzy_inplace(file_text, search, replace, min_ratio)?;
    unified_diff(file_text, &updated, path)
}

pub fn apply_fuzzy_inplace(
    file_text: &str,
    search: &str,
    replace: &str,
    min_ratio: f32,
) -> Option<String> {
    if search.trim().is_empty() {
        return None;
    }
    if let Some(pos) = file_text.find(search) {
        let mut out = String::new();
        out.push_str(&file_text[..pos]);
        out.push_str(replace);
        out.push_str(&file_text[pos + search.len()..]);
        return Some(out);
    }

    let src_lines: Vec<&str> = file_text.split_inclusive('\n').collect();
    let src_no_nl: Vec<String> = src_lines
        .iter()
        .map(|l| l.trim_end_matches('\n').to_string())
        .collect();
    let src_stripped: Vec<String> = src_no_nl.iter().map(|l| l.trim().to_string()).collect();

    let search_lines = strip_lines(search);
    let replace_lines = strip_lines(replace);
    let search_stripped: Vec<String> = search_lines.iter().map(|l| l.trim().to_string()).collect();
    let n = search_stripped.len();
    if n == 0 || n > src_stripped.len() {
        return None;
    }

    let mut hit: Option<(usize, usize)> = None;
    for i in 0..=src_stripped.len() - n {
        if src_stripped[i..i + n] == search_stripped[..] {
            hit = Some((i, i + n));
            break;
        }
    }

    if hit.is_none() {
        let mut best_i = usize::MAX;
        let mut best_r = 0.0f32;
        let mut best_win = n;
        let win_lo = n.saturating_sub(2).max(1);
        let win_hi = (n + 3).min(src_stripped.len());
        for win in win_lo..=win_hi {
            if win > src_stripped.len() {
                break;
            }
            for i in 0..=src_stripped.len() - win {
                let window = &src_stripped[i..i + win];
                let r = sequence_ratio(&search_stripped, window);
                if r > best_r {
                    best_r = r;
                    best_i = i;
                    best_win = win;
                }
            }
        }
        if best_r < min_ratio || best_i == usize::MAX {
            return None;
        }
        hit = Some((best_i, best_i + best_win));
    }

    let (start, end) = hit.unwrap();
    let mut base_indent = String::new();
    for j in start..end {
        if !src_no_nl[j].trim().is_empty() {
            base_indent = leading_ws(&src_no_nl[j]);
            break;
        }
    }
    let mut search_base = String::new();
    for ln in &search_lines {
        if !ln.trim().is_empty() {
            search_base = leading_ws(ln);
            break;
        }
    }
    let mut src_indents: Vec<String> = (0..replace_lines.len())
        .map(|i| {
            if start + i < end {
                leading_ws(&src_no_nl[start + i])
            } else {
                base_indent.clone()
            }
        })
        .collect();
    if !src_indents.is_empty() && replace_lines.len() > (end - start) {
        let last = src_indents[src_indents.len().min(end - start).saturating_sub(1)].clone();
        for i in (end - start)..replace_lines.len() {
            src_indents[i] = last.clone();
        }
    }
    let new_body = reindent_replace(&replace_lines, &src_indents, &base_indent, &search_base);
    let mut out = String::new();
    for line in &src_lines[..start] {
        out.push_str(line);
    }
    for (i, ln) in new_body.iter().enumerate() {
        out.push_str(ln);
        let need_nl = i + 1 < new_body.len()
            || (end > start
                && end - 1 < src_lines.len()
                && src_lines[end - 1].ends_with('\n'));
        if need_nl && !ln.ends_with('\n') {
            out.push('\n');
        }
    }
    for line in &src_lines[end..] {
        out.push_str(line);
    }
    if out == file_text {
        return None;
    }
    Some(out)
}

/// Single-line substring replace (GGUF indent drift).
pub fn apply_line_substr_replace(file_text: &str, search: &str, replace: &str) -> Option<String> {
    let s_lines: Vec<&str> = search.lines().filter(|l| !l.trim().is_empty()).collect();
    let r_lines: Vec<&str> = replace.lines().filter(|l| !l.trim().is_empty()).collect();
    if s_lines.len() != 1 || r_lines.len() != 1 {
        return None;
    }
    let needle = s_lines[0].trim();
    let rep = r_lines[0].trim();
    if needle.len() < 8 {
        return None;
    }
    let mut out = String::new();
    let mut changed = false;
    for ln in file_text.split_inclusive('\n') {
        let has_nl = ln.ends_with('\n');
        let core = ln.trim_end_matches('\n');
        if !changed && core.trim().contains(needle) {
            let indent = leading_ws(core);
            let new_core = if core.trim() == needle {
                format!("{indent}{rep}")
            } else {
                core.replacen(needle, rep, 1)
            };
            out.push_str(&new_core);
            if has_nl {
                out.push('\n');
            }
            changed = true;
        } else {
            out.push_str(ln);
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}
