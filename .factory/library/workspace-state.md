## Workspace state notes

- During feature `t7-mlb-lineup-client`, the working tree contained unrelated local weather/test WIP.
- To keep the lineup commit clean, these repo-local stashes were created:
  - `worker-8eb1cd35 isolate unrelated weather WIP`
  - `worker-8eb1cd35 isolate unrelated test scaffolding`
  - `worker-8eb1cd35 isolate leftover weather test`
- Reapply with `git stash list` / `git stash pop` only if that unrelated weather work needs to be resumed.
