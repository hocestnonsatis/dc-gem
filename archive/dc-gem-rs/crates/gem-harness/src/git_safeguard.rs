//! Phase-18: Git Safety & Rollback Guard for autonomous `dc-gem fix`.
//!
//! Flow:
//! 1. Snapshot HEAD (+ optional stash of dirty tree)
//! 2. Create ephemeral branch `dc-gem/auto-fix-<ts>`
//! 3. Apply patch → verify → optional self-repair
//! 4. Success: commit on work branch
//! 5. Failure: `git reset --hard <start>` + restore original ref + drop work branch

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};

/// Outcome of a safeguarded fix session.
#[derive(Clone, Debug)]
pub struct SafeguardReport {
    pub work_branch: String,
    pub start_sha: String,
    pub original_ref: String,
    pub committed: bool,
    pub rolled_back: bool,
    pub verify_ok: bool,
    pub message: String,
}

/// Active git snapshot for one fix session.
pub struct GitSafeguard {
    repo: PathBuf,
    start_sha: String,
    original_ref: String,
    work_branch: String,
    stash_pushed: bool,
    finished: bool,
}

/// Resolve the git work-tree root for `path` (walks parents via `rev-parse`).
pub fn resolve_git_root(path: &Path) -> Result<PathBuf> {
    let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(&path)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .context("git rev-parse --show-toplevel")?;
    if !out.status.success() {
        bail!(
            "not a git repository: {}\n\
             Hint: run once with `--init-git` to create an initial snapshot, or:\n\
               cd {} && git init && git add -A && git commit -m \"init\"\n\
             Or disable the guard: `--no-git-safeguard`",
            path.display(),
            path.display()
        );
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(PathBuf::from(root))
}

/// `git init` + initial commit so the safeguard has a HEAD to snapshot.
pub fn init_git_repo(path: &Path) -> Result<PathBuf> {
    let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    eprintln!("[Git Safeguard] initializing git repo at {}", path.display());
    let out = Command::new("git")
        .args(["init"])
        .current_dir(&path)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .context("git init")?;
    if !out.status.success() {
        bail!("git init failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    // Ensure a basic ignore file exists (do not overwrite user file)
    let gi = path.join(".gitignore");
    if !gi.exists() {
        fs::write(
            &gi,
            "target/\n.venv/\n__pycache__/\n*.gguf\nmodels/\ndata/swe_repos/\nlogs/\n",
        )?;
    }
    let _ = git(
        &path,
        &[
            "-c",
            "user.email=dc-gem@localhost",
            "-c",
            "user.name=dc-gem",
            "add",
            "-A",
        ],
    )?;
    let out = git(
        &path,
        &[
            "-c",
            "user.email=dc-gem@localhost",
            "-c",
            "user.name=dc-gem",
            "commit",
            "-m",
            "dc-gem: initial snapshot for git safeguard",
        ],
    )?;
    if !out.status.success() {
        // empty commit allowed if nothing staged
        let status = git_stdout(&path, &["status", "--porcelain"]).unwrap_or_default();
        if status.trim().is_empty() {
            let out2 = git(
                &path,
                &[
                    "-c",
                    "user.email=dc-gem@localhost",
                    "-c",
                    "user.name=dc-gem",
                    "commit",
                    "--allow-empty",
                    "-m",
                    "dc-gem: initial snapshot for git safeguard",
                ],
            )?;
            if !out2.status.success() {
                bail!(
                    "git commit failed: {}",
                    String::from_utf8_lossy(&out2.stderr)
                );
            }
        } else {
            bail!(
                "git commit failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
    resolve_git_root(&path)
}

impl GitSafeguard {
    /// Begin a session: stash dirty tree if needed, create ephemeral branch.
    pub fn begin(repo: &Path) -> Result<Self> {
        let repo = resolve_git_root(repo)?;

        // Need at least one commit for HEAD
        let head_ok = git(&repo, &["rev-parse", "--verify", "HEAD"])
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !head_ok {
            bail!(
                "git repo has no commits yet: {}\n\
                 Hint: `git commit --allow-empty -m init` or re-run with `--init-git`",
                repo.display()
            );
        }

        let start_sha = git_stdout(&repo, &["rev-parse", "HEAD"])?.trim().to_string();
        let original_ref = current_ref(&repo)?;

        let mut stash_pushed = false;
        if !worktree_clean(&repo)? {
            eprintln!("[Git Safeguard] dirty tree → stash push -u");
            let out = git(&repo, &["stash", "push", "-u", "-m", "dc-gem-safeguard-auto"])?;
            if !out.status.success() {
                bail!(
                    "git stash failed: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            stash_pushed = true;
        }

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let work_branch = format!("dc-gem/auto-fix-{ts}");

        eprintln!("[Git Safeguard] creating work branch {work_branch} @ {start_sha}");
        let out = git(&repo, &["checkout", "-b", &work_branch])?;
        if !out.status.success() {
            // restore stash before failing
            if stash_pushed {
                let _ = git(&repo, &["stash", "pop"]);
            }
            bail!(
                "cannot create branch {work_branch}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        Ok(Self {
            repo,
            start_sha,
            original_ref,
            work_branch,
            stash_pushed,
            finished: false,
        })
    }

    pub fn repo(&self) -> &Path {
        &self.repo
    }

    pub fn work_branch(&self) -> &str {
        &self.work_branch
    }

    pub fn start_sha(&self) -> &str {
        &self.start_sha
    }

    /// Apply a unified diff to the worktree (`git apply`).
    pub fn apply_patch(&self, patch: &str) -> Result<()> {
        if patch.trim().is_empty() {
            bail!("empty patch");
        }
        let tmp = self.repo.join(".dc_gem_safeguard_apply.diff");
        fs::write(&tmp, patch).context("write patch tempfile")?;
        let out = git(
            &self.repo,
            &[
                "apply",
                "--whitespace=nowarn",
                tmp.to_str().unwrap_or(".dc_gem_safeguard_apply.diff"),
            ],
        )?;
        let _ = fs::remove_file(&tmp);
        if !out.status.success() {
            bail!(
                "git apply failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        Ok(())
    }

    /// Hard-reset worktree to the session start SHA (keeps work branch).
    pub fn reset_to_start(&self) -> Result<()> {
        let out = git(&self.repo, &["reset", "--hard", &self.start_sha])?;
        if !out.status.success() {
            bail!(
                "git reset --hard failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        let _ = git(&self.repo, &["clean", "-fd"]);
        Ok(())
    }

    /// Run a shell verify command in the repo; returns (ok, combined output).
    pub fn run_verify(&self, verify_cmd: &str) -> Result<(bool, String)> {
        let out = Command::new("bash")
            .arg("-lc")
            .arg(verify_cmd)
            .current_dir(&self.repo)
            .output()
            .context("spawn verify_cmd")?;
        let mut text = String::new();
        text.push_str(&String::from_utf8_lossy(&out.stdout));
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        Ok((out.status.success(), text))
    }

    /// Stage all and commit on the work branch.
    pub fn commit_success(&mut self, message: &str) -> Result<()> {
        let _ = git(&self.repo, &["add", "-A"])?;
        // Allow empty? No — if nothing staged, fail softly
        let staged = git_stdout(&self.repo, &["diff", "--cached", "--name-only"])?;
        if staged.trim().is_empty() {
            // maybe only unstaged — try add again already done; check worktree diff
            let dirty = git_stdout(&self.repo, &["status", "--porcelain"])?;
            if dirty.trim().is_empty() {
                bail!("nothing to commit after successful verify");
            }
            let _ = git(&self.repo, &["add", "-A"])?;
        }
        let msg = if message.is_empty() {
            "dc-gem: autonomous fix verified & applied"
        } else {
            message
        };
        let out = git(
            &self.repo,
            &[
                "-c",
                "user.email=dc-gem@localhost",
                "-c",
                "user.name=dc-gem",
                "commit",
                "-m",
                msg,
            ],
        )?;
        if !out.status.success() {
            bail!(
                "git commit failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        self.finished = true;
        Ok(())
    }

    /// Rollback to start SHA, restore original ref, delete work branch, pop stash.
    pub fn rollback(mut self) -> Result<SafeguardReport> {
        eprintln!(
            "[Git Safeguard] ROLLBACK → {} (drop {})",
            self.start_sha, self.work_branch
        );
        let _ = git(&self.repo, &["reset", "--hard", &self.start_sha]);
        let _ = git(&self.repo, &["clean", "-fd"]);

        // Return to original branch / detached SHA
        let checkout_target = if self.original_ref.starts_with("refs/")
            || (!self.original_ref.is_empty() && self.original_ref != "HEAD")
        {
            // Prefer branch name without refs/heads/
            self.original_ref
                .strip_prefix("refs/heads/")
                .unwrap_or(&self.original_ref)
                .to_string()
        } else {
            self.start_sha.clone()
        };
        let _ = git(&self.repo, &["checkout", &checkout_target]);

        // Delete work branch if it still exists and we are not on it
        let _ = git(&self.repo, &["branch", "-D", &self.work_branch]);

        if self.stash_pushed {
            let _ = git(&self.repo, &["stash", "pop"]);
            self.stash_pushed = false;
        }

        self.finished = true;
        Ok(SafeguardReport {
            work_branch: self.work_branch.clone(),
            start_sha: self.start_sha.clone(),
            original_ref: self.original_ref.clone(),
            committed: false,
            rolled_back: true,
            verify_ok: false,
            message: "rolled back to start snapshot".into(),
        })
    }

    /// Mark session successful (commit already done); drop stash bookkeeping.
    pub fn finish_success(mut self) -> Result<SafeguardReport> {
        if self.stash_pushed {
            // Keep stash entry? User's pre-fix dirty state was stashed under the
            // original ref. Leave it for manual `stash pop` on original branch.
            eprintln!(
                "[Git Safeguard] note: pre-fix dirty state remains in stash (dc-gem-safeguard-auto)"
            );
            self.stash_pushed = false;
        }
        self.finished = true;
        Ok(SafeguardReport {
            work_branch: self.work_branch.clone(),
            start_sha: self.start_sha.clone(),
            original_ref: self.original_ref.clone(),
            committed: true,
            rolled_back: false,
            verify_ok: true,
            message: format!("verified & committed on {}", self.work_branch),
        })
    }
}

impl Drop for GitSafeguard {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Emergency rollback if caller forgot to finish
        eprintln!("[Git Safeguard] Drop without finish → emergency reset");
        let _ = git(&self.repo, &["reset", "--hard", &self.start_sha]);
        let _ = git(&self.repo, &["clean", "-fd"]);
        let target = self
            .original_ref
            .strip_prefix("refs/heads/")
            .unwrap_or(&self.original_ref);
        let _ = git(&self.repo, &["checkout", target]);
        let _ = git(&self.repo, &["branch", "-D", &self.work_branch]);
        if self.stash_pushed {
            let _ = git(&self.repo, &["stash", "pop"]);
        }
    }
}

fn git(repo: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .args(args)
        .current_dir(repo)
        .env("GIT_TERMINAL_PROMPT", "0")
        .output()
        .with_context(|| format!("git {}", args.join(" ")))
}

fn git_stdout(repo: &Path, args: &[&str]) -> Result<String> {
    let out = git(repo, args)?;
    if !out.status.success() {
        return Err(anyhow!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

fn worktree_clean(repo: &Path) -> Result<bool> {
    let s = git_stdout(repo, &["status", "--porcelain"])?;
    Ok(s.trim().is_empty())
}

fn current_ref(repo: &Path) -> Result<String> {
    // Prefer symbolic-ref; fall back to SHA when detached
    let out = git(repo, &["symbolic-ref", "--short", "-q", "HEAD"])?;
    if out.status.success() {
        return Ok(String::from_utf8_lossy(&out.stdout).trim().to_string());
    }
    git_stdout(repo, &["rev-parse", "HEAD"]).map(|s| s.trim().to_string())
}

/// Apply → verify → on fail reset + retry closure (1×) → commit or full rollback.
pub fn execute_with_safeguard<F>(
    repo: &Path,
    verify_cmd: &str,
    mut attempt: F,
) -> Result<SafeguardReport>
where
    F: FnMut(usize, Option<&str>) -> Result<String>,
{
    let guard = GitSafeguard::begin(repo)?;
    let mut last_err_out = String::new();

    for round in 0..2 {
        guard.reset_to_start()?;
        let patch = match attempt(round, if round == 0 { None } else { Some(last_err_out.as_str()) })
        {
            Ok(p) => p,
            Err(e) => {
                last_err_out = format!("generate failed: {e}");
                continue;
            }
        };
        if patch.trim().is_empty() {
            last_err_out = "empty patch".into();
            continue;
        }
        if let Err(e) = guard.apply_patch(&patch) {
            last_err_out = format!("apply failed: {e}");
            continue;
        }
        eprintln!(
            "[Git Safeguard] verify round {} : {}",
            round + 1,
            verify_cmd
        );
        let (ok, out) = guard.run_verify(verify_cmd)?;
        if ok {
            let mut g = guard;
            g.commit_success("dc-gem: autonomous fix verified & applied")?;
            return g.finish_success();
        }
        last_err_out = out;
        eprintln!(
            "[Git Safeguard] verify failed (round {}):\n{}",
            round + 1,
            last_err_out.chars().take(600).collect::<String>()
        );
    }

    guard.rollback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn init_repo(dir: &Path) {
        assert!(Command::new("git")
            .args(["init"])
            .current_dir(dir)
            .status()
            .unwrap()
            .success());
        // silence git identity for commit
        let _ = Command::new("git")
            .args(["config", "user.email", "t@t"])
            .current_dir(dir)
            .status();
        let _ = Command::new("git")
            .args(["config", "user.name", "t"])
            .current_dir(dir)
            .status();
        fs::write(dir.join("hello.txt"), "hello\n").unwrap();
        assert!(Command::new("git")
            .args(["add", "hello.txt"])
            .current_dir(dir)
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir)
            .status()
            .unwrap()
            .success());
    }

    #[test]
    fn safeguard_commits_on_verify_ok() {
        let dir = tempdir().unwrap();
        init_repo(dir.path());
        let report = execute_with_safeguard(dir.path(), "test -f hello.txt", |_round, _err| {
            let patch = "\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+hello world
";
            Ok(patch.into())
        })
        .unwrap();
        assert!(report.committed);
        assert!(report.verify_ok);
        assert!(!report.rolled_back);
        let body = fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert!(body.contains("hello world"));
    }

    #[test]
    fn safeguard_rolls_back_on_verify_fail() {
        let dir = tempdir().unwrap();
        init_repo(dir.path());
        let start = fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        let report = execute_with_safeguard(dir.path(), "false", |_round, _err| {
            let patch = "\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+broken
";
            Ok(patch.into())
        })
        .unwrap();
        assert!(report.rolled_back);
        assert!(!report.committed);
        let body = fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(body, start);
    }

    #[test]
    fn safeguard_self_repair_second_round() {
        let dir = tempdir().unwrap();
        init_repo(dir.path());
        let report = execute_with_safeguard(dir.path(), "grep -q 'fixed' hello.txt", |round, _| {
            if round == 0 {
                Ok("\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+wrong
"
                .into())
            } else {
                Ok("\
diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-hello
+fixed
"
                .into())
            }
        })
        .unwrap();
        assert!(report.committed);
        let body = fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert!(body.contains("fixed"));
    }
}
