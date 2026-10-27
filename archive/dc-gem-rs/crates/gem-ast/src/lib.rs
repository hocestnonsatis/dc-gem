//! Multi-language AST chunking (tree-sitter) and 4-stage patch salvager.

mod chunk;
mod fuzzy;
mod salvage;

pub use chunk::{chunk_file_by_symbols, SymbolChunk};
pub use fuzzy::{apply_search_replace_fuzzy, unified_diff};
pub use salvage::{
    extract_plus_minus_edits, parse_loose_search_replace, replace_ast_symbol_in_file,
    salvage_and_apply_patch, PatchError, SalvageMeta,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_python_and_rust_symbols() {
        let py = "class A:\n    def foo(self):\n        return 1\n\ndef bar():\n    return 2\n";
        let chunks = chunk_file_by_symbols("mod.py", py, 200);
        assert!(chunks.iter().any(|c| c.symbol == "A" || c.symbol == "A.foo"));
        assert!(chunks.iter().any(|c| c.symbol == "bar"));
        assert!(chunks[0].text.contains("# [FILE: mod.py | SYMBOL:"));

        let rs = "struct Foo {}\nimpl Foo {\n    fn bar(&self) {}\n}\nfn baz() {}\n";
        let chunks = chunk_file_by_symbols("lib.rs", rs, 200);
        assert!(chunks.iter().any(|c| c.symbol.contains("Foo") || c.symbol == "baz"));
    }

    #[test]
    fn fuzzy_search_replace_broken_indent() {
        let src = "class Widget:\n    def render(self, x):\n        return x - 1\n";
        let diff = apply_search_replace_fuzzy(
            src,
            "return x - 1",
            "return x + 1",
            "pkg/w.py",
            0.72,
        )
        .expect("fuzzy apply");
        assert!(diff.contains("diff --git"));
        assert!(diff.contains("return x + 1"));
    }

    #[test]
    fn salvage_loose_original_updated() {
        let src = "class Widget:\n    def render(self, x):\n        return x - 1\n\n    def other(self):\n        return 0\n";
        let raw = "Original:\n        return x - 1\nUpdated:\n        return x + 1\n";
        let (patch, meta) = salvage_and_apply_patch("pkg/w.py", src, raw).unwrap();
        assert_eq!(meta.stage, Some(1));
        assert!(patch.contains("return x + 1"));
    }

    #[test]
    fn salvage_plus_minus_and_broken_header() {
        let src = "class Widget:\n    def render(self, x):\n        return x - 1\n";
        let raw = "diff --gita/pkg/w.py b/pkg/w.py\n--- a/pkg/w.py\n+++ b/pkg/w.py\n@@ -1,3 +1,3 @@\n class Widget:\n     def render(self, x):\n-        return x - 1\n+        return x + 1\n";
        let (patch, meta) = salvage_and_apply_patch("pkg/w.py", src, raw).unwrap();
        assert!(meta.stage == Some(1) || meta.stage == Some(2));
        assert!(patch.contains("return x + 1"));
    }

    #[test]
    fn replace_ast_method_body() {
        let src = "class Widget:\n    def render(self, x):\n        return x - 1\n\n    def other(self):\n        return 0\n";
        let new_def = "def render(self, x):\n    return x + 1\n";
        let diff = replace_ast_symbol_in_file(src, new_def, "pkg/w.py", Some("render"))
            .expect("ast replace");
        assert!(diff.contains("diff --git"));
        assert!(diff.contains("return x + 1"));
    }

    #[test]
    fn salvage_stage3_markdown_method() {
        let src = "class Widget:\n    def render(self, x):\n        return x - 1\n\n    def other(self):\n        return 0\n";
        let raw = "Here is the fixed method:\n```python\ndef render(self, x):\n    return x + 1\n```\n";
        let (patch, meta) = salvage_and_apply_patch("pkg/w.py", src, raw).unwrap();
        assert_eq!(meta.stage, Some(3));
        assert!(patch.contains("return x + 1"));
    }

    #[test]
    fn extract_pm_edits() {
        let raw = "-opts = cur_model._meta\n+opts = cur_model._meta.concrete_model._meta\n";
        let edits = extract_plus_minus_edits(raw);
        assert_eq!(edits.len(), 1);
        assert!(edits[0].0.contains("cur_model._meta"));
        assert!(edits[0].1.contains("concrete_model"));
    }

    #[test]
    fn parse_loose_sr_blocks() {
        let raw = "Original:\n        return x - 1\nUpdated:\n        return x + 1\n";
        let edits = parse_loose_search_replace(raw);
        assert!(edits.iter().any(|e| e.1.contains("x - 1") && e.2.contains("x + 1")));
    }
}
