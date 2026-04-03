# MLB Run Model Research Assumptions

This file is the repo-local MLB research reference for the away-run roadmap in `docs/roadmaps/AWAY_RUN_RESEARCH_5_STAGE_PLAN.md`.

## Important Repo Note

- `docs/research/ResearchInformation.md` is the pointer to the MLB modeling docs and Stage 5 dual-view outputs.
- The old unrelated Micro-SaaS material was archived to `archive/research/ResearchInformation.micro_saas.md`.
- Use this file and the roadmap when the task is about away-run modeling.

## Accepted Research Assumptions

1. Single-game `R^2` is a weak north-star for betting. Distribution quality matters more than point accuracy alone.
2. MLB run counts are overdispersed and zero-heavy. Mean-only Poisson views are not enough.
3. Sequencing matters enough that a Markov or Monte Carlo lane is worth building as a separate research path.
4. Shrinkage and calibration matter and should be reused before any large trainer rewrite.
5. Market priors matter and should eventually be added as a distinct research lane rather than silently blended into the control.
6. Air density, catcher framing, and umpire effects are real, but they should be expressed in betting-relevant outputs.

## Concise Rationale

- Distribution scoring:
  - Betting decisions depend on tails, shutout risk, and interval quality. A good mean can still imply a bad run distribution.

- Overdispersion and zero mass:
  - MLB away-run outcomes have more variance and more zeros than a simple Poisson mean view explains. Stage 2 and Stage 3 should measure and model that directly.

- Market priors:
  - Market prices contain useful information about team strength, context, and injuries. They belong in a separate, auditable research lane.

- Sequencing and MCMC:
  - Expected runs alone cannot represent inning state, cluster luck, or base-out transitions. A slower simulation lane is justified as a second viewpoint.

## Reuse-First Guidance

- Reuse current shrinkage from `src/features/marcel_blend.py`.
- Reuse current weather logic from `src/features/adjustments/weather.py`.
- Reuse current framing logic from `src/features/defense.py`.
- Reuse current umpire history features from `src/features/umpires.py`.
- Do not treat Stage 1 as permission to replace the control trainer.
