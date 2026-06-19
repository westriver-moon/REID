# Git Provenance

This workspace is managed as a single Git repository rooted at:

`/home/cgv841/ybj`

The nested Git metadata below was intentionally moved out of the workspace on
2026-06-19 during the single-repository unification.

## Former Nested Repositories

| Path | Former remote | Captured branch | Captured HEAD |
| --- | --- | --- | --- |
| `TVI-LFM` | `https://github.com/WHU-HZY/TVI-LFM.git` | `main` | `682742130f2fb7bca26dabd92bc5a788225d7541` |
| `Single-experiment/external/Self-Correction-Human-Parsing` | `https://github.com/GoGoDuck912/Self-Correction-Human-Parsing.git` | `master` | `eb84c432cc697f494d99662a05f2335eb2f26095` |

## Backup Location

The moved Git directories and metadata snapshots are stored outside the
workspace at:

`/home/cgv841/git-backups/ybj-git-unify-20260619-225549/`

This backup includes each former nested repository's remote, HEAD, branch, and
status output captured before the `.git` directories were moved.

## Notes

- `TVI-LFM` is now tracked as ordinary files by the outer workspace repository.
- `Single-experiment/external/Self-Correction-Human-Parsing` was converted from
  a gitlink/submodule-style entry to ordinary files.
- Large runtime artifacts such as checkpoints, logs, datasets, arrays, and
  model weights remain ignored by the outer repository.
