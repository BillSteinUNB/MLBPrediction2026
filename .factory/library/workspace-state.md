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

- During feature `fix-t8-weather-cache-ttl`, pre-existing local `src/model/data_builder.py` and `tests/test_data_builder.py` WIP surfaced while validating the weather cache fix.
- `tests/test_data_builder.py` broke full-suite pytest collection because `src.model.data_builder` is not part of the committed milestone yet.
- To run milestone validators cleanly without committing unrelated WIP, these repo-local stashes were created:
  - `worker-0cac70ef isolate unrelated data_builder test`
  - `worker-0cac70ef isolate unrelated data_builder WIP`

- During feature `fix-t7-lineup-date-scoping`, another repo-local stash was created to isolate unrelated training-data work before validation:
  - `worker-ba06611d isolate unrelated data_builder WIP`

- During feature `fix-t6-cache-invalidation`, another repo-local stash was created to isolate unrelated model/training work before validation:
  - `worker-4991f8d1 isolate unrelated model WIP`
