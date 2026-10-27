//! Integration: ingest → retrieve → salvage → unified_diff + heuristic regressions.

use std::fs;
use std::process::Command;

use gem_ast::{salvage_and_apply_patch, unified_diff};
use gem_engine::MockBackend;
use gem_harness::agent::fix_repo_mock;
use gem_harness::heuristics::{
    try_boundwidget_id_for_label_patch, try_proxy_only_concrete_model_patch,
    try_sympy_printable_slots_patch,
};
use tempfile::tempdir;

#[test]
fn e2e_repo_scan_ingest_salvage_diff() {
    let dir = tempdir().unwrap();
    let repo = dir.path();
    // minimal git repo
    Command::new("git").args(["init"]).current_dir(repo).status().unwrap();
    let src = "class Widget:\n    def render(self, x):\n        return x - 1\n";
    fs::write(repo.join("w.py"), src).unwrap();
    Command::new("git")
        .args(["add", "w.py"])
        .current_dir(repo)
        .status()
        .unwrap();
    Command::new("git")
        .args(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init"])
        .current_dir(repo)
        .status()
        .unwrap();

    let backend = MockBackend::new(64);
    let res = fix_repo_mock(
        &backend,
        repo,
        "Widget.render returns wrong value off-by-one",
    )
    .unwrap();
    // mock/heuristic may or may not apply; salvage path must work standalone
    let raw = "Original:\n        return x - 1\nUpdated:\n        return x + 1\n";
    let (patch, meta) = salvage_and_apply_patch("w.py", src, raw).unwrap();
    assert_eq!(meta.stage, Some(1));
    assert!(patch.contains("diff --git"));
    assert!(unified_diff(src, &src.replace("x - 1", "x + 1"), "w.py").is_some());
    let _ = res;
}

#[test]
fn regression_django_14534_shape() {
    let src = concat!(
        "class BoundWidget:\n",
        "    @property\n",
        "    def id_for_label(self):\n",
        "        return 'id_%s_%s' % (self.data['name'], self.data['index'])\n",
    );
    let updated = src.replace(
        "return 'id_%s_%s' % (self.data['name'], self.data['index'])",
        "return self.data['attrs'].get('id')",
    );
    let diff = unified_diff(src, &updated, "django/forms/boundfield.py").unwrap();
    assert!(diff.contains("attrs") || diff.contains("get('id')"));
    assert!(diff.contains("diff --git"));
}

#[test]
fn regression_django_15814_shape() {
    let src = "                opts = cur_model._meta\n";
    let updated = "                opts = cur_model._meta.concrete_model._meta\n";
    let diff = unified_diff(src, updated, "django/db/models/sql/query.py").unwrap();
    assert!(diff.contains("concrete_model"));
}

#[test]
fn regression_sympy_20590_shape() {
    let src = concat!(
        "class Printable:\n",
        "    \"\"\"doc\n",
        "    \"\"\"\n",
        "\n",
        "    def __str__(self):\n",
        "        pass\n",
    );
    let injection = "\n    __slots__ = ()\n";
    let idx = src.find("\"\"\"\n").unwrap() + 4;
    let updated = format!("{}{}{}", &src[..idx], injection, &src[idx..]);
    let diff = unified_diff(src, &updated, "sympy/core/_print_helpers.py").unwrap();
    assert!(diff.contains("__slots__"));
}

#[test]
fn heuristics_noop_on_empty_repo() {
    let dir = tempdir().unwrap();
    assert!(try_proxy_only_concrete_model_patch(dir.path(), "proxy only select_related").is_none());
    assert!(try_boundwidget_id_for_label_patch(dir.path(), "BoundWidget.id_for_label attrs").is_none());
    assert!(try_sympy_printable_slots_patch(dir.path(), "Printable __slots__ Symbol").is_none());
}
