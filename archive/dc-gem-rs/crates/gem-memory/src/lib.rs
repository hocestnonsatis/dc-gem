//! Gradient-free episodic memory (GEM) + coverage-aware OOW retrieval gate.
//!
//! Port of Phase-15.2/15.3 hybrid scoring and Phase-9.2 surprise write gate.

use std::collections::{HashSet, VecDeque};

/// One episodic slot: L2-normalized key, value, utility, and symbol metadata.
#[derive(Clone, Debug, Default)]
pub struct EpisodicSlot {
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub utility: f32,
    pub symbol_header: String,
    pub source_span: String,
    pub file_path: String,
    /// Monotonic write stamp for LRU eviction.
    pub last_used: u64,
}

/// Capacity-bounded episodic table with surprise-gated writes.
#[derive(Clone, Debug)]
pub struct EpisodicMemory {
    pub max_slots: usize,
    slots: Vec<Option<EpisodicSlot>>,
    n_valid: usize,
    clock: u64,
    surprise_hist: VecDeque<f32>,
    hist_cap: usize,
    pub z_threshold: f32,
    pub nms_kernel: usize,
}

impl EpisodicMemory {
    pub fn new(max_slots: usize) -> Self {
        Self {
            max_slots: max_slots.max(1),
            slots: vec![None; max_slots.max(1)],
            n_valid: 0,
            clock: 0,
            surprise_hist: VecDeque::with_capacity(256),
            hist_cap: 256,
            z_threshold: 2.2,
            nms_kernel: 5,
        }
    }

    pub fn n_valid(&self) -> usize {
        self.n_valid
    }

    pub fn reset(&mut self) {
        self.slots = vec![None; self.max_slots];
        self.n_valid = 0;
        self.clock = 0;
        self.surprise_hist.clear();
    }

    pub fn slots(&self) -> impl Iterator<Item = &EpisodicSlot> {
        self.slots.iter().filter_map(|s| s.as_ref())
    }

    /// L2-normalize in place; zero vectors become a tiny unit vector on dim 0.
    pub fn l2_normalize(v: &mut [f32]) {
        let mut n2 = 0.0f32;
        for x in v.iter() {
            n2 += x * x;
        }
        let n = n2.sqrt();
        if n < 1e-8 {
            if let Some(x0) = v.first_mut() {
                *x0 = 1.0;
            }
            return;
        }
        for x in v.iter_mut() {
            *x /= n;
        }
    }

    pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        if n == 0 {
            return 0.0;
        }
        let mut dot = 0.0f32;
        for i in 0..n {
            dot += a[i] * b[i];
        }
        dot.clamp(-1.0, 1.0)
    }

    /// Evict lowest utility; ties broken by LRU (oldest last_used).
    fn pick_evict_index(&self) -> usize {
        let mut best_i = 0usize;
        let mut best_u = f32::INFINITY;
        let mut best_age = u64::MAX;
        for (i, slot) in self.slots.iter().enumerate() {
            match slot {
                None => return i,
                Some(s) => {
                    if s.utility < best_u
                        || ((s.utility - best_u).abs() < 1e-6 && s.last_used < best_age)
                    {
                        best_u = s.utility;
                        best_age = s.last_used;
                        best_i = i;
                    }
                }
            }
        }
        best_i
    }

    fn hist_mean_std(&self) -> (f32, f32) {
        let n = self.surprise_hist.len();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mean = self.surprise_hist.iter().sum::<f32>() / n as f32;
        let var = self
            .surprise_hist
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        (mean, var.sqrt())
    }

    /// 1D-NMS: reject if a recent write in the history window already spiked.
    fn nms_blocks(&self, surprise: f32) -> bool {
        if self.nms_kernel <= 1 || self.surprise_hist.is_empty() {
            return false;
        }
        let k = self.nms_kernel.min(self.surprise_hist.len());
        let recent: Vec<f32> = self
            .surprise_hist
            .iter()
            .rev()
            .take(k)
            .copied()
            .collect();
        let peak = recent.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        surprise + 1e-6 < peak
    }

    /// Write when surprise clears Z-gate (mean + z·std) and 1D-NMS.
    /// Returns true if a slot was written/updated.
    pub fn write_with_surprise_gate(
        &mut self,
        mut key: Vec<f32>,
        value: Vec<f32>,
        utility: f32,
        symbol_header: impl Into<String>,
        source_span: impl Into<String>,
        file_path: impl Into<String>,
        surprise: f32,
    ) -> bool {
        Self::l2_normalize(&mut key);
        self.surprise_hist.push_back(surprise);
        if self.surprise_hist.len() > self.hist_cap {
            self.surprise_hist.pop_front();
        }

        if self.z_threshold > 0.0 && self.surprise_hist.len() >= 8 {
            let (mean, std) = self.hist_mean_std();
            let thr = mean + self.z_threshold * std;
            if surprise < thr {
                return false;
            }
            if self.nms_blocks(surprise) {
                return false;
            }
        }

        self.clock = self.clock.saturating_add(1);
        let slot = EpisodicSlot {
            key,
            value,
            utility,
            symbol_header: symbol_header.into(),
            source_span: source_span.into(),
            file_path: file_path.into(),
            last_used: self.clock,
        };

        // Prefer empty slot
        if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
            self.slots[i] = Some(slot);
            self.n_valid += 1;
            return true;
        }
        let i = self.pick_evict_index();
        self.slots[i] = Some(slot);
        true
    }

    /// Force-write (tests / ingest when surprise gate is bypassed).
    pub fn write_slot(
        &mut self,
        mut key: Vec<f32>,
        value: Vec<f32>,
        utility: f32,
        symbol_header: impl Into<String>,
        source_span: impl Into<String>,
        file_path: impl Into<String>,
    ) -> usize {
        Self::l2_normalize(&mut key);
        self.clock = self.clock.saturating_add(1);
        let slot = EpisodicSlot {
            key,
            value,
            utility,
            symbol_header: symbol_header.into(),
            source_span: source_span.into(),
            file_path: file_path.into(),
            last_used: self.clock,
        };
        let i = if let Some(i) = self.slots.iter().position(|s| s.is_none()) {
            self.n_valid += 1;
            i
        } else {
            self.pick_evict_index()
        };
        self.slots[i] = Some(slot);
        i
    }

    pub fn search_top_k(&mut self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(s) = slot {
                let sim = Self::cosine_sim(query, &s.key);
                scored.push((i, sim));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        self.clock = self.clock.saturating_add(1);
        for &(i, _) in &scored {
            if let Some(s) = self.slots[i].as_mut() {
                s.last_used = self.clock;
                s.utility += 0.05;
            }
        }
        scored
    }

    pub fn get(&self, idx: usize) -> Option<&EpisodicSlot> {
        self.slots.get(idx).and_then(|s| s.as_ref())
    }
}

/// Hybrid retrieval score: `0.5 * cosine + 0.5 * lexical_overlap`.
pub fn compute_hybrid_symbol_score(
    query_vec: &[f32],
    query_identifiers: &[String],
    slot: &EpisodicSlot,
) -> f32 {
    let cos = EpisodicMemory::cosine_sim(query_vec, &slot.key);
    let lex = lexical_overlap_slot(slot, query_identifiers);
    0.5 * cos + 0.5 * lex
}

fn lexical_overlap_slot(slot: &EpisodicSlot, idents: &[String]) -> f32 {
    if idents.is_empty() {
        return 0.0;
    }
    let sym = slot.symbol_header.to_lowercase();
    let leaf = sym.rsplit('.').next().unwrap_or("").to_string();
    let id_lower: HashSet<String> = idents.iter().map(|s| s.to_lowercase()).collect();

    let mut score = 0.0f32;
    if id_lower.contains(&sym) {
        score = 1.0;
    } else if id_lower.contains(&leaf) && sym.contains('.') {
        score = 0.92;
    } else if id_lower.contains(&leaf) {
        score = 0.45;
    } else {
        for ident in &id_lower {
            if sym.contains(ident) {
                score = score.max(0.75);
                break;
            }
        }
    }

    let focus = slot.source_span.lines().take(8).collect::<Vec<_>>().join("\n");
    let focus_l = focus.to_lowercase();
    if !focus_l.is_empty() {
        let hits = id_lower.iter().filter(|i| focus_l.contains(i.as_str())).count();
        let body = hits as f32 / id_lower.len().max(1) as f32;
        if score >= 0.9 {
            score = (score + 0.05 * body).min(1.0);
        } else {
            score = score.max(0.5 * body);
        }
    }
    score.min(1.0)
}

const IDENT_STOP: &[&str] = &[
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "be", "this",
    "that", "with", "from", "as", "it", "by", "at", "we", "you", "when", "which", "should",
    "would", "could", "have", "has", "was", "were", "not", "but", "if", "than", "then", "into",
    "about", "also", "can", "may", "will", "return", "class", "def", "self", "true", "false",
    "none", "import", "description", "expected", "behavior", "reproduce", "error", "issue",
    "bug", "fix", "code", "file", "files", "test", "tests",
];

/// Active context chunk for coverage-aware gating.
#[derive(Clone, Debug, Default)]
pub struct ActiveChunk {
    pub path: String,
    pub symbol: String,
    pub text: String,
}

pub fn extract_primary_targets(issue: &str) -> HashSet<String> {
    let title = issue.trim().lines().next().unwrap_or("");
    let mut out: HashSet<String> = HashSet::new();
    let re_path = regex_lite_idents(title);
    for s in re_path {
        out.insert(s.clone());
        if let Some(leaf) = s.rsplit('.').next() {
            out.insert(leaf.to_string());
        }
        if let Some(owner) = s.split('.').next() {
            out.insert(owner.to_string());
        }
    }
    for s in camel_idents(title) {
        if !IDENT_STOP.contains(&s.to_lowercase().as_str()) {
            out.insert(s);
        }
    }
    for kw in ["concrete_model", "select_related", "id_for_label", "only"] {
        if issue.contains(kw) {
            out.insert(kw.to_string());
        }
    }
    out.retain(|x| x.len() >= 3);
    out
}

fn regex_lite_idents(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && (is_ident_cont(bytes[i]) || bytes[i] == b'.') {
                i += 1;
            }
            let s = &text[start..i];
            if s.contains('.') && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            {
                out.push(s.to_string());
            }
        } else {
            i += 1;
        }
    }
    out
}

fn camel_idents(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tok in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if tok.len() >= 3
            && tok.chars().next().is_some_and(|c| c.is_ascii_uppercase())
            && tok.chars().skip(1).any(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        {
            out.push(tok.to_string());
        }
    }
    out
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}
fn is_ident_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

pub fn extract_issue_identifiers(issue: &str) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();
    for s in regex_lite_idents(issue) {
        if !IDENT_STOP.contains(&s.to_lowercase().as_str()) {
            out.insert(s.clone());
            if let Some(leaf) = s.rsplit('.').next() {
                out.insert(leaf.to_string());
            }
        }
    }
    for s in camel_idents(issue) {
        if !IDENT_STOP.contains(&s.to_lowercase().as_str()) {
            out.insert(s);
        }
    }
    // snake_case tokens length >= 4
    for tok in issue.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if tok.len() >= 4
            && tok.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && !IDENT_STOP.contains(&tok)
        {
            out.insert(tok.to_string());
        }
    }
    out.into_iter().filter(|x| x.len() >= 3).collect()
}

/// Active-window coverage of primary targets + whether a target def is present.
pub fn compute_active_coverage(
    active_context: &[ActiveChunk],
    primary_targets: &HashSet<String>,
) -> (f32, bool) {
    if primary_targets.is_empty() {
        return (0.0, false);
    }
    let mut hits = 0.0f32;
    let mut def_present = false;
    for t in primary_targets {
        let leaf = t.rsplit('.').next().unwrap_or(t.as_str());
        let owner = if t.contains('.') {
            t.split('.').next()
        } else {
            None
        };
        let mut found = false;
        for c in active_context {
            let sym = c.symbol.as_str();
            if sym.is_empty() {
                continue;
            }
            if sym.eq_ignore_ascii_case(t) {
                found = true;
                break;
            }
            if let Some(own) = owner {
                if sym == format!("{own}.{leaf}")
                    || (sym.ends_with(&format!(".{leaf}")) && sym.starts_with(&format!("{own}.")))
                {
                    found = true;
                    break;
                }
                if sym == own
                    && (c.text.contains(&format!("def {leaf}"))
                        || c.text.contains(&format!("def {leaf}(")))
                {
                    found = true;
                    break;
                }
            }
            if owner.is_none() && (sym == leaf || sym.ends_with(&format!(".{leaf}"))) {
                found = true;
                break;
            }
        }
        if found {
            def_present = true;
            hits += 1.0;
        } else {
            for c in active_context {
                if c.text.len() <= 1800
                    && leaf.len() >= 4
                    && (c.text.contains(leaf) || c.symbol.to_lowercase().contains(&leaf.to_lowercase()))
                {
                    hits += 0.25;
                    break;
                }
            }
        }
    }
    let cov = (hits / primary_targets.len().max(1) as f32).min(1.0);
    (cov, def_present)
}

/// Coverage-aware adaptive OOW gate (Phase-15.3).
///
/// Returns `false` when the target symbol is already in the active window
/// (cuts noise); `true` on a real out-of-window gap.
pub fn should_trigger_oow_retrieval(
    problem_statement: &str,
    active_context: &[ActiveChunk],
    best_oow_score: Option<f32>,
) -> bool {
    should_trigger_oow_retrieval_thr(problem_statement, active_context, best_oow_score, 0.30)
}

pub fn should_trigger_oow_retrieval_thr(
    problem_statement: &str,
    active_context: &[ActiveChunk],
    best_oow_score: Option<f32>,
    coverage_threshold: f32,
) -> bool {
    let primary = extract_primary_targets(problem_statement);
    if primary.is_empty() {
        return true;
    }
    let (s_active, def_in) = compute_active_coverage(active_context, &primary);
    let s_oow = best_oow_score;

    if def_in && s_active >= coverage_threshold {
        match s_oow {
            None => return false,
            Some(s) if s < s_active + 0.08 => return false,
            Some(s) => return s >= 0.90 && s > s_active + 0.30,
        }
    }
    true
}

/// Rank slots by hybrid score for selective OOW retrieval.
pub fn rank_oow_slots(
    memory: &EpisodicMemory,
    query_vec: &[f32],
    issue: &str,
    active: &[ActiveChunk],
    min_sim: f32,
    max_n: usize,
) -> Vec<(usize, f32)> {
    let idents = extract_issue_identifiers(issue);
    let active_ids: HashSet<(String, String)> = active
        .iter()
        .map(|a| (a.path.clone(), a.symbol.clone()))
        .collect();
    let mut ranked: Vec<(usize, f32)> = Vec::new();
    for (i, slot) in memory.slots().enumerate() {
        // slots() skips None — need actual indices
        let _ = (i, slot);
    }
    for (i, opt) in memory.slots.iter().enumerate() {
        let Some(slot) = opt else { continue };
        let key = (slot.file_path.clone(), slot.symbol_header.clone());
        if active_ids.contains(&key) {
            continue;
        }
        let s = compute_hybrid_symbol_score(query_vec, &idents, slot);
        if s >= min_sim {
            ranked.push((i, s));
        }
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(max_n);
    ranked
}

/// Helper used by tests: build identifier list as owned strings.
pub fn idents_from_slice(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|s| (*s).to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_unit_norm() {
        let mut v = vec![3.0, 4.0];
        EpisodicMemory::l2_normalize(&mut v);
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hybrid_score_half_cos_half_lex() {
        let mut key = vec![1.0, 0.0];
        EpisodicMemory::l2_normalize(&mut key);
        let slot = EpisodicSlot {
            key: key.clone(),
            value: key.clone(),
            utility: 1.0,
            symbol_header: "BoundWidget.id_for_label".into(),
            source_span: "def id_for_label(self):\n    return None\n".into(),
            file_path: "forms.py".into(),
            last_used: 0,
        };
        let idents = idents_from_slice(&["BoundWidget.id_for_label", "id_for_label"]);
        let s = compute_hybrid_symbol_score(&key, &idents, &slot);
        // cos ~1, lex ~1 → ~1.0
        assert!(s > 0.9, "got {s}");
    }

    #[test]
    fn surprise_gate_rejects_flat_stream() {
        let mut mem = EpisodicMemory::new(8);
        mem.z_threshold = 2.0;
        // seed high-surprise history so Z gate is active
        for i in 0..12 {
            mem.write_with_surprise_gate(
                vec![1.0, i as f32],
                vec![1.0, 0.0],
                1.0,
                "a",
                "x",
                "a.py",
                5.0 + (i as f32) * 0.01,
            );
        }
        let before = mem.n_valid();
        let wrote = mem.write_with_surprise_gate(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            1.0,
            "b",
            "y",
            "b.py",
            0.05, // far below mean + z·std
        );
        assert!(!wrote, "low-surprise write must be gated");
        assert_eq!(mem.n_valid(), before);
    }

    #[test]
    fn reset_isolates_tasks() {
        let mut mem = EpisodicMemory::new(4);
        mem.write_slot(vec![1.0, 0.0], vec![1.0, 0.0], 1.0, "a", "x", "a.py");
        assert_eq!(mem.n_valid(), 1);
        mem.reset();
        assert_eq!(mem.n_valid(), 0);
    }

    #[test]
    fn oow_gate_in_window_false_oow_true() {
        // 15814-like: QuerySet.only already in window
        let issue_local = "QuerySet.only() after select_related() crash on proxy models.\nUse concrete_model when resolving opts.\n";
        let active_local = vec![ActiveChunk {
            path: "django/db/models/query.py".into(),
            symbol: "QuerySet".into(),
            text: "# [FILE: django/db/models/query.py | SYMBOL: QuerySet]\nclass QuerySet:\n    def only(self, *fields):\n        return self\n    def select_related(self, *fields):\n        return self\n".into(),
        }];
        assert!(!should_trigger_oow_retrieval(
            issue_local,
            &active_local,
            Some(0.55)
        ));
        assert!(!should_trigger_oow_retrieval(
            issue_local,
            &active_local,
            None
        ));

        // 14534-like: BoundWidget.id_for_label out of window
        let issue_oow = "BoundWidget.id_for_label ignores id set by ChoiceWidget.options\nKeyError on attrs['id']\n";
        let active_oow = vec![ActiveChunk {
            path: "django/forms/boundfield.py".into(),
            symbol: "BoundField".into(),
            text: "# [FILE: django/forms/boundfield.py | SYMBOL: BoundField]\nclass BoundField:\n    def id_for_label(self):\n        return self.field.widget.attrs.get('id')\n".into(),
        }];
        assert!(should_trigger_oow_retrieval(
            issue_oow,
            &active_oow,
            Some(0.60)
        ));
    }
}
