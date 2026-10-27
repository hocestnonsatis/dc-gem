//! `dc-gem` CLI — fix / swe-bench.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use gem_engine::{accel_label, GgufBackend, GgufConfig, MockBackend};
use gem_harness::agent::{fix_repo, fix_repo_safeguarded, FixRequest};
use gem_harness::git_safeguard::{init_git_repo, resolve_git_root};
use gem_harness::swe::{run_swe_bench, SweBenchArgs};

#[derive(Parser, Debug)]
#[command(name = "dc-gem", version, about = "DC-GEM Rust coding agent")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Live coding agent: ingest → OOW → salvage → optional verify + git safeguard
    Fix {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long)]
        issue: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long = "verify-cmd")]
        verify_cmd: Option<String>,
        /// Enable Git snapshot / verify / commit / rollback (default: on when --verify-cmd set)
        #[arg(long, default_value_t = true)]
        git_safeguard: bool,
        /// Disable Git safeguard even if --verify-cmd is set
        #[arg(long = "no-git-safeguard", action = clap::ArgAction::SetTrue)]
        no_git_safeguard: bool,
        /// If `--repo` is not a git work tree, `git init` + initial commit first
        #[arg(long = "init-git", action = clap::ArgAction::SetTrue)]
        init_git: bool,
        #[arg(long, default_value = "true")]
        mem_on: bool,
        #[arg(long, default_value_t = 8)]
        max_files: usize,
        /// Use deterministic mock backend (CI)
        #[arg(long)]
        mock: bool,
    },
    /// SWE-bench Lite batch evaluator
    SweBench {
        #[arg(long)]
        subset: PathBuf,
        #[arg(long, default_value = "memON")]
        variant: String,
        #[arg(long, default_value = "predictions_rust.jsonl")]
        output: PathBuf,
        #[arg(long, default_value = "../data/swe_repos")]
        repo_cache: PathBuf,
        #[arg(long, default_value = "dc-gem-rs-ternary")]
        model_name: String,
        #[arg(long)]
        skip_checkout: bool,
        #[arg(long)]
        mock: bool,
        #[arg(long)]
        model: Option<String>,
    },
}

fn build_backend(mock: bool, model: Option<&str>) -> Result<Box<dyn gem_engine::CoreBackend>> {
    if mock {
        return Ok(Box::new(MockBackend::new(256)));
    }
    let mut cfg = GgufConfig::default();
    if let Some(m) = model {
        let p = PathBuf::from(m);
        if p.is_file() {
            cfg.model_path = Some(p);
        } else if m.starts_with("http://") || m.starts_with("https://") {
            cfg.server_url = m.to_string();
        }
    }
    eprintln!(
        "[backend] gguf accel={} server={} model={:?}",
        accel_label(),
        cfg.server_url,
        cfg.model_path
    );
    Ok(Box::new(GgufBackend::new(cfg)))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::Fix {
            repo,
            issue,
            model,
            verify_cmd,
            git_safeguard,
            no_git_safeguard,
            init_git,
            mem_on,
            max_files,
            mock,
        } => {
            let backend = build_backend(mock, model.as_deref())?;
            let use_guard = verify_cmd.is_some() && git_safeguard && !no_git_safeguard;
            let mut repo = repo;
            if use_guard || init_git {
                match resolve_git_root(&repo) {
                    Ok(root) => {
                        eprintln!("[Git Safeguard] git root: {}", root.display());
                        // Keep agent scan root as user requested; safeguard uses toplevel.
                        let _ = root;
                    }
                    Err(_) if init_git => {
                        let root = init_git_repo(&repo)?;
                        eprintln!("[Git Safeguard] initialized at {}", root.display());
                        repo = root;
                    }
                    Err(e) => return Err(e),
                }
            }
            let req = FixRequest {
                repo,
                issue,
                verify_cmd: verify_cmd.clone(),
                mem_on,
                max_files,
                extra_paths: Vec::new(),
            };
            let res = if use_guard {
                eprintln!("[Git Safeguard] enabled (ephemeral branch + verify/rollback)");
                fix_repo_safeguarded(backend.as_ref(), &req)?
            } else {
                fix_repo(backend.as_ref(), &req)?
            };
            println!("method={}", res.method);
            println!("applied_ok={}", res.applied_ok);
            println!("gem_hit={}", res.gem_hit);
            if let Some(v) = res.verify_ok {
                println!("verify_ok={v}");
            }
            if let Some(sg) = &res.safeguard {
                println!("safeguard.committed={}", sg.committed);
                println!("safeguard.rolled_back={}", sg.rolled_back);
                println!("safeguard.work_branch={}", sg.work_branch);
                println!("safeguard.message={}", sg.message);
            }
            println!("--- patch ---");
            println!("{}", res.patch);
        }
        Commands::SweBench {
            subset,
            variant,
            output,
            repo_cache,
            model_name,
            skip_checkout,
            mock,
            model,
        } => {
            let backend = build_backend(mock, model.as_deref())?;
            let metrics = run_swe_bench(
                backend.as_ref(),
                &SweBenchArgs {
                    subset,
                    variant,
                    output,
                    repo_cache,
                    model_name,
                    skip_checkout,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
    }
    Ok(())
}
