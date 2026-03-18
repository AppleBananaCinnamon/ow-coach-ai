# Killfeed Pipeline Handoff

## Current Goal
Reduce duplicate kill events without losing true kills.

## Big Picture
- Recall is much better than it was earlier.
- Arrow detection is effectively solved.
- The remaining main problem is duplicate suppression.
- There is also a separate parser-validation issue where some real kills survive detector dedupe but fail to enter `parsed_events_valid.json`.

## What Is Working
- Arrow detection is very strong.
  - Uses asset template matching.
  - Central-band arrow search was a good change.
- Parent ROI alignment between detector and subregion tuning is aligned.
- Best-frame selection improved when we biased toward structurally complete kill rows.
- Persisting `right_icon_box` from live detection was a good change.
  - Re-detecting the arrow later from saved JPEGs was unreliable.
  - Dedupe now uses stored geometry instead of re-finding the arrow.
- Current system can capture all 4 true kills in some runs.

## What Is Not Solved
- Duplicate kills still survive into `parsed_events_valid.json`.
- Some real kills survive `detections_deduped.json` but fail parser validation.
- Clustering is still a bottleneck:
  - it can merge distinct nearby kills before dedupe gets a chance to distinguish them.

## Current Important Files
- [killfeed_live_detector.py](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\killfeed_live_detector.py)
- [killfeed_parser.py](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\killfeed_parser.py)
- [arrow_detector.py](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\arrow_detector.py)

## Current Pipeline Understanding
1. Detector triggers on killfeed ROI change.
2. Burst candidates are captured.
3. A `best` frame is selected from the burst.
4. Raw detections are clustered.
5. Clustered detections are deduped.
6. Parser validates and emits:
   - `parsed_events.json`
   - `parsed_events_valid.json`
   - `parsed_events_merged.json`

## Key Learnings

### 1. Arrow is the right geometric anchor
- Name lengths shift the row horizontally.
- Arrow-relative subregions are the correct mental model.
- Clipping computed subregions to ROI bounds is necessary.

### 2. Motion score is useful for triggering, not for best-frame selection
- High motion often corresponds to transitional / rollout frames.
- Stable later frames can be better representatives even with lower motion.
- Motion was removed from the last tiebreak in `best_candidate` selection.

### 3. Right-icon-only dedupe is not enough in its current form
- Using the victim icon as the primary duplicate signal is directionally correct.
- But raw grayscale icon fingerprints are still noisy.
- Earlier failures came from trying to re-detect the arrow on saved crops.
- That was improved by persisting `right_icon_box`.
- Even after that, duplicates still survive.

### 4. Clustering can destroy true kills before dedupe
- Example: `killfeed_evt00005_best_f0000035_t00003.58.jpg`
  - Survived raw detection.
  - Was absorbed during clustering because it was only `0.401s` from the next event.
  - Cluster representative was chosen by `motion_score * signal`.
- This means clustering is still too blunt.

### 5. Parser validation is separate from dedupe
- If a crop is in `detections_deduped.json` but not in `parsed_events_valid.json`, it fell out in parser validation.
- Parser currently mostly rejects with `missing_left_or_right_icon`.
- `parsed_events.json` now includes `reject_reason`.

## Recent Code Changes That Should Be Preserved

### Detector
- Central-band arrow search.
- Best-frame selection now prefers:
  - arrow present
  - both icons fully visible
  - both icons visible
  - both nameplates visible
  - combined icon coverage
  - signal as weak final tiebreak
- `right_icon_box` is stored on:
  - `BurstCandidate`
  - `FeedDetection`
- Dedupe now uses stored `right_icon_box` instead of re-detecting arrow from saved crops.
- Dedupe currently compares against all prior kept events within a 5-second window, not just the most recent event in a row.

### Parser
- `parsed_events.json` now includes `reject_reason`.

## Debug Artifacts Available
- [artifacts/killfeed_live/best_selection_debug.jsonl](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\artifacts\killfeed_live\best_selection_debug.jsonl)
  - Shows why a burst candidate won or lost.
- [artifacts/killfeed_live/dedupe_debug.jsonl](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\artifacts\killfeed_live\dedupe_debug.jsonl)
  - Shows dedupe comparisons, similarity, threshold, and merge result.
- [artifacts/killfeed_live/right_icon_debug](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\artifacts\killfeed_live\right_icon_debug)
  - Exports accepted/rejected right-icon crops used by dedupe.
- [artifacts/killfeed_live/parsed_events.json](C:\Users\jmfra\OneDrive\Desktop\Coding\ow_coach\artifacts\killfeed_live\parsed_events.json)
  - Now includes parser reject reasons.

## Proven Failure Modes
- Duplicates can survive because:
  - right-icon similarity threshold is too strict for some same-victim duplicates
  - some duplicates were never compared under older dedupe logic
  - clustering can collapse distinct kills before dedupe
- Real kills can be lost because:
  - clustering absorbs them into a nearby event
  - parser rejects them due to missing icon recovery

## Recommended Next Steps

### Highest Priority
1. Rework clustering so it is not purely temporal.
   - Time should define which events are candidates for comparison.
   - Identity should decide whether events merge.
   - Current clustering is still destructive.

2. Improve icon representation for dedupe.
   - Current grayscale fingerprint may be too noisy.
   - Best next heuristic experiment:
     - masked / binary icon fingerprinting
     - or stronger normalization of icon crops

### Secondary
3. Investigate parser rejects using new `reject_reason`.
   - Specifically inspect real kills present in `detections_deduped.json` but missing from `parsed_events_valid.json`.

4. Consider slightly higher `sample-fps`.
   - If only one usable frame of a kill appears in the funnel, this may help more than increasing `burst-count`.

## Ideas Discussed But Not Yet Implemented
- Asset-based hero icon classification.
  - Could improve both dedupe and hero recognition.
  - Should likely happen after icon crop stability is better.
- Lightweight ML classifier on icon crops.
  - Promising medium-term direction.
- Multi-frame confirmation within an event window.
  - Could help distinguish same kill vs new kill more robustly.

## Important Conceptual Takeaway
The system does best when:
- motion opens the door
- arrow anchors geometry
- identity-bearing UI regions decide what is truly new

The system does poorly when:
- motion is used to choose representative frames
- time alone decides clustering
- noisy whole-crop similarity stands in for identity
