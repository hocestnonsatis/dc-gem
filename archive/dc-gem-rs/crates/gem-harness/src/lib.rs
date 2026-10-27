//! Autonomous coding agent + SWE-bench Lite harness.

pub mod agent;
pub mod git_safeguard;
pub mod heuristics;
pub mod swe;

pub use agent::{fix_repo, fix_repo_safeguarded, FixRequest, FixResult};
pub use git_safeguard::{
    execute_with_safeguard, init_git_repo, resolve_git_root, GitSafeguard, SafeguardReport,
};
pub use swe::{run_swe_bench, SweBenchArgs, SweMetrics};
