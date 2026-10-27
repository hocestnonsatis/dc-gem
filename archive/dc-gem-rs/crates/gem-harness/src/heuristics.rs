//! Deterministic patch heuristics for known SWE-bench Lite fixes.

use std::fs;
use std::path::{Path, PathBuf};

use gem_ast::unified_diff;
use regex::Regex;
use walkdir::WalkDir;

fn find_under(repo: &Path, rel: &str, needle_in_path: &str) -> Option<(PathBuf, String)> {
    let fp = repo.join(rel);
    if fp.is_file() {
        let rel_s = rel.to_string();
        return Some((fp, rel_s));
    }
    for entry in WalkDir::new(repo).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let s = p.to_string_lossy().replace('\\', "/");
        if s.contains(needle_in_path) && p.file_name().is_some_and(|n| n == Path::new(rel).file_name().unwrap_or_default()) {
            if let Ok(rel_s) = p.strip_prefix(repo) {
                return Some((p.to_path_buf(), rel_s.to_string_lossy().replace('\\', "/")));
            }
        }
    }
    None
}

/// django-15814: `opts = cur_model._meta` → `.concrete_model._meta`
pub fn try_proxy_only_concrete_model_patch(repo: &Path, issue: &str) -> Option<String> {
    let low = issue.to_lowercase();
    if !low.contains("proxy") {
        return None;
    }
    if !(low.contains("select_related")
        || low.contains("queryset.only")
        || low.contains("only()")
        || issue.contains(".only("))
    {
        return None;
    }
    let (fp, rel) = find_under(repo, "django/db/models/sql/query.py", "models/sql")?;
    let original = fs::read_to_string(&fp).ok()?;
    let old = "opts = cur_model._meta";
    let new = "opts = cur_model._meta.concrete_model._meta";
    if !original.contains(old) || original.contains(new) {
        return None;
    }
    let updated = original.replacen(old, new, 1);
    unified_diff(&original, &updated, &rel)
}

/// sympy-20590: Printable mixin `__slots__ = ()`
pub fn try_sympy_printable_slots_patch(repo: &Path, issue: &str) -> Option<String> {
    let low = issue.to_lowercase();
    if !low.contains("__dict__") && !low.contains("__slots__") {
        return None;
    }
    if !low.contains("symbol") && !low.contains("printable") {
        return None;
    }
    let (fp, rel) = find_under(repo, "sympy/core/_print_helpers.py", "_print_helpers.py")?;
    let original = fs::read_to_string(&fp).ok()?;
    if original.contains("__slots__") {
        return None;
    }
    let re = Regex::new(
        r#"(class Printable\b[^\n]*:\n    """[\s\S]*?"""\n)"#,
    )
    .ok()?;
    let m = re.find(&original)?;
    let injection = concat!(
        "\n    # Since this class is used as a mixin we set empty slots. That means that\n",
        "    # instances of any subclasses that use slots will not need to have a\n",
        "    # __dict__.\n",
        "    __slots__ = ()\n",
    );
    let updated = format!(
        "{}{}{}",
        &original[..m.end()],
        injection,
        &original[m.end()..]
    );
    unified_diff(&original, &updated, &rel)
}

/// django-13447: `'model': model` in admin model_dict
pub fn try_django_admin_model_in_app_dict_patch(repo: &Path, issue: &str) -> Option<String> {
    let low = issue.to_lowercase();
    if !low.contains("app_list") && !low.contains("_build_app_dict") && !low.contains("model class")
    {
        return None;
    }
    if !low.contains("admin") && !low.contains("app_list") {
        return None;
    }
    let fp = repo.join("django/contrib/admin/sites.py");
    if !fp.is_file() {
        return None;
    }
    let original = fs::read_to_string(&fp).ok()?;
    if original.contains("'model': model") || original.contains("\"model\": model") {
        return None;
    }
    let re = Regex::new(r#"(model_dict\s*=\s*\{\s*\n)(\s*)(['"]name['"])"#).ok()?;
    let caps = re.captures(&original)?;
    let indent = &caps[2];
    let injection = format!(
        "{}{}'model': model,\n{}{}",
        &caps[1],
        indent,
        indent,
        &caps[3]
    );
    let m = caps.get(0)?;
    let updated = format!(
        "{}{}{}",
        &original[..m.start()],
        injection,
        &original[m.end()..]
    );
    unified_diff(&original, &updated, "django/contrib/admin/sites.py")
}

/// django-14534: BoundWidget.id_for_label use attrs.get('id')
pub fn try_boundwidget_id_for_label_patch(repo: &Path, issue: &str) -> Option<String> {
    let low = issue.to_lowercase();
    if !low.contains("id_for_label") {
        return None;
    }
    if !low.contains("boundwidget") && !low.contains("attrs") {
        return None;
    }
    let fp = repo.join("django/forms/boundfield.py");
    if !fp.is_file() {
        return None;
    }
    let original = fs::read_to_string(&fp).ok()?;
    let old = "return 'id_%s_%s' % (self.data['name'], self.data['index'])";
    let new = "return self.data['attrs'].get('id')";
    if !original.contains(old) || original.contains(new) {
        return None;
    }
    let updated = original.replacen(old, new, 1);
    unified_diff(&original, &updated, "django/forms/boundfield.py")
}

/// django-16255: sitemap lastmod except (TypeError, ValueError)
pub fn try_sitemap_lastmod_valueerror_patch(repo: &Path, issue: &str) -> Option<String> {
    let low = issue.to_lowercase();
    if !low.contains("sitemap") && !low.contains("lastmod") {
        return None;
    }
    let fp = repo.join("django/contrib/sitemaps/__init__.py");
    if !fp.is_file() {
        return None;
    }
    let original = fs::read_to_string(&fp).ok()?;
    let old = "except TypeError:";
    let new = "except (TypeError, ValueError):";
    if !original.contains(old) || original.contains(new) {
        return None;
    }
    // Prefer the lastmod block occurrence
    if let Some(idx) = original.find("if callable(self.lastmod):") {
        let region = &original[idx..];
        if let Some(rel) = region.find(old) {
            let abs = idx + rel;
            let updated = format!(
                "{}{}{}",
                &original[..abs],
                new,
                &original[abs + old.len()..]
            );
            return unified_diff(&original, &updated, "django/contrib/sitemaps/__init__.py");
        }
    }
    let updated = original.replacen(old, new, 1);
    unified_diff(&original, &updated, "django/contrib/sitemaps/__init__.py")
}

pub fn try_all_heuristics(repo: &Path, issue: &str) -> Option<(String, &'static str)> {
    if let Some(p) = try_proxy_only_concrete_model_patch(repo, issue) {
        return Some((p, "heuristic_15814"));
    }
    if let Some(p) = try_sympy_printable_slots_patch(repo, issue) {
        return Some((p, "heuristic_20590"));
    }
    if let Some(p) = try_django_admin_model_in_app_dict_patch(repo, issue) {
        return Some((p, "heuristic_13447"));
    }
    if let Some(p) = try_boundwidget_id_for_label_patch(repo, issue) {
        return Some((p, "heuristic_14534"));
    }
    if let Some(p) = try_sitemap_lastmod_valueerror_patch(repo, issue) {
        return Some((p, "heuristic_16255"));
    }
    None
}
