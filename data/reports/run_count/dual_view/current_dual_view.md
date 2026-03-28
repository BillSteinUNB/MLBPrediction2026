# Run Count Dual View

Generated at: `2026-03-28T19:34:22.915867+00:00`

## Lanes

| Lane | Status | Expected away runs | Shutout | P(away >= 3) | P(away >= 5) |
| --- | --- | ---: | ---: | ---: | ---: |
| Control mean lane | control | 4.410 | 0.070 | 0.680 | 0.407 |
| Best distribution lane | promoted_second_opinion | 4.344 | 0.071 | 0.678 | 0.406 |
| Best MCMC lane | exploratory | 3.936 | 0.171 | 0.532 | 0.320 |

## Promotion

- Best current research lane: `Best distribution lane`
- Promoted second-opinion lane: `Best distribution lane`
- Production-promotable lane: `None`
- The best research lane is promoted only as a second opinion until betting evidence is sufficient.
- No research lane is production-promotable under the current Stage 5 rule set.

## Disagreements

| Pair | Mean gap | Shutout gap | P>=3 gap | P>=5 gap | Any flag |
| --- | ---: | ---: | ---: | ---: | --- |
| Control mean lane vs Best distribution lane | 0.066 | 0.002 | 0.002 | 0.001 | no |
| Control mean lane vs Best MCMC lane | 0.474 | 0.101 | 0.149 | 0.087 | yes |
| Best distribution lane vs Best MCMC lane | 0.408 | 0.100 | 0.146 | 0.086 | yes |
