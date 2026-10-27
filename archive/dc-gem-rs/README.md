# dc-gem-rs (Phase-17)

Zero-Python DC-GEM stack: episodic memory, tree-sitter AST salvager, GGUF engine, and SWE-bench harness.

## Crates

| Crate | Role |
|-------|------|
| `gem-memory` | Gradient-free episodic slots + hybrid score + coverage-aware OOW gate |
| `gem-ast` | tree-sitter Python/Rust chunking + 4-stage patch salvager |
| `gem-engine` | `CoreBackend` — GGUF/`llama-server` + deterministic mock |
| `gem-harness` | `dc-gem` CLI (`fix`, `swe-bench`) |

## Feature flags

```toml
default = ["cpu"]
# --features rocm|cuda|metal|vulkan
```

GPU features select llama `-ngl` offload profile (ROCm/HIP `gfx1201`, CUDA, Metal, Vulkan).

## Build & test

```bash
cd dc-gem-rs
cargo test --workspace
cargo run --release -- swe-bench \
  --subset ../examples/swe_lite_subset.json \
  --variant memON \
  --repo-cache ../data/swe_repos \
  --output ../runs/swe_dc_gem/predictions_rust.jsonl
```

Live agent (Phase-18 Git Safeguard — default on with `--verify-cmd`):

```bash
cargo run --release -- fix \
  --repo . \
  --issue "BoundWidget.id_for_label KeyError" \
  --verify-cmd "cargo test --workspace -q"
# success → commit on dc-gem/auto-fix-<ts>
# failure → reset --hard to pre-fix snapshot + drop work branch
```

Disable guard: `--no-git-safeguard`.

Environment: `DC_GEM_LLAMA_SERVER_URL` (default `http://127.0.0.1:8765`).
