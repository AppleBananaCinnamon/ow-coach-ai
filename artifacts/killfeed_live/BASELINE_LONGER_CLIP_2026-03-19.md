# Killfeed Baseline: Longer Clip (2026-03-19)

This run is the current longer-clip reference baseline for the killfeed pipeline.

## Outcome

- Actual kills in clip: 5
- Canonical kill events detected: 5
- False positives (Type I): 0
- False negatives (Type II): 0
- Result: perfect kill-event mapping for this longer validation sequence

## Branch Context

- Branch: `codex/killfeed-persistence-window-exp`
- Key change vs the earlier baseline:
  - dedupe now models longer killfeed persistence windows
  - same-row persistence can survive longer when victim identity remains strong
  - adjacent-row persistence is also modeled more explicitly

## Notes

- This baseline should be compared against:
  - [BASELINE_2026-03-19.md](/C:/Users/jmfra/OneDrive/Desktop/Coding/ow_coach/artifacts/killfeed_live/BASELINE_2026-03-19.md)
- The earlier baseline validated the shorter clip at `4/4`.
- This longer-clip baseline validated the longer sequence at `5/5`.
- Together, these give us:
  - one shorter control case
  - one longer persistence-sensitive case

## Recommendation

- Keep the earlier perfect short-clip baseline branch intact.
- Keep this longer-clip result as the current best candidate for integration.
- Continue evaluating on additional clips before treating the logic as fully general.
