## Workspace state notes

- During feature `t7-mlb-lineup-client`, the working tree contained unrelated local weather/test WIP.
- To keep the lineup commit clean, these repo-local stashes were created:
  - `worker-8eb1cd35 isolate unrelated weather WIP`
  - `worker-8eb1cd35 isolate unrelated test scaffolding`
  - `worker-8eb1cd35 isolate leftover weather test`
- Reapply with `git stash list` / `git stash pop` only if that unrelated weather work needs to be resumed.

- During feature `fix-t5-odds-api-commit-games`, additional repo-local stashes were created to isolate unrelated feature-engineering and local note WIP:
  - `worker-ea65a6a9 isolate unrelated feature-engineering WIP retry`
  - `worker-ea65a6a9 isolate reserved nul path`
  - `worker-ea65a6a9 isolate unrelated feature-engineering WIP`
- A stray untracked `nul` path required Git Bash removal (`rm -f`) after Windows git operations hit reserved-path handling; if `git status` ever shows `?? nul` again, inspect it from Bash under `/mnt/c/...` instead of standard Windows file APIs.
