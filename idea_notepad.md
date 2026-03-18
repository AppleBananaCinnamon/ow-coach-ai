1. Hybrid
   1. round 1 hybrid time = 4 minutes
   2. fight starts around 3:55
2. Flashpoint
   1. 30 second delay before match start
   2. 40 second interval between conquering a flashpoint
3. Overall
   1. 10s respawn on death
   2. How would you categorize a fight where there is one death, but no others? 
4. Zen Info
   1. Ult last ~5 secs
   2. Zen start charging right click immediately on first sight of enemy
   3. Always a few meters from cover
   4. Takes perks ascendence and focused destruction
   5. Ult charge gained per fight 

Learnings
1. We broke out ULT and POSITION because these are state identifiers -- while it's useful to know what the outcome is (e.g., death), knowing how the player got there is also important. 
   1. This is similar to user funnel journey metrics -- purchase and attrition are not the only things that matter; decomposing how they got there usually matter more.
   2. It also makes CV ingestion cleaner later - e.g., at frame 3, Zen had 44% ult; at frame 10, Zen was on high-ground; on frame 12, Zen killed Anran. 
   3. Those are not deaths. They are observations of state. If an attribute is intrinsic to that event (i.e., 'ult % at death'), then we use it as an inline. If the attribute is observable state that can exist outside the event (e.g., high-ground), make it it's own event.
2. Only mark IMPORTANT events in our system. Otherwise, you're just tracking move up, down, pitch aim left, right. What defines an important event? If it has a direct impat on the game's outcome. 
3. Manual annotation is untenable. For 9 minutes of gameplay, even trying to capture the key events took roughly 3.5 hours. This was even after converting annotations to a more human-friendly workbook in Excel. 
   1. Important: almost all other ML projects start with human annotation as the ground truth; it's not the tediousness which makes this redundant in our case but the fact that the killfeed and hero UI gives us almost everything we need; a machine would be better suited and tracking those diffs than a human, that is why we pivoted.
4. Machine vision (CV) isn't without it's own set of problems. Tuning the right killfeed box, trying to avoid the color flair from someone popping off and going on fire, trying not to make the box too wide or too tall to ingest unncessary images (the background will always change as the character is moving).
5. After implementing a burst shot approach, we improved the fidelity of the extraction but are now picking up extra noise from the color sampling. We're going to implement a combined gate heuristic: 1) it should filter out low saturation pixel changes (i.e., what we believe is just the environs); 2) it will look for explicit high color UI bars.
   1. We can attach a binary classifier at some point (i.e., killfeed_present vs. not_killfeed_present),
   2. Core idea: 
   3. ```
        Only trigger a killfeed event if:
            diff_score is high
        AND the crop contains enough saturated pixels
        ```
6. We are picking up too much noise from the burst approach, and the generic color palette change isn't helping much. We implemented a structural detection module; that means it's looking for a specific color change within a defined geometric space to avoid picking up just whenever the background changed. 
7. On 3.11, switched to Codex for making in-line changes. Will need to find a way to review changes before implementing them so I can ensure I'm still learning along the way, that was the early advantage of the 'manual patch' approach.
   1. Codex is very helpful! Removes the problem with stitching in code piecemeal, and now I can abstract the actual tuning changes unless they are one-line tweaks to parameters.
8. Main challenge is getting true events from the killfeed that aren't overrepresented. The pixel change is just enough that it's throwing off me detecting events; still, making progress from actually picking up the kill. Now I just need to isolate out the noise.
9. 3.12 -- after spending 90 more mins trying to fiddle the grayscale and color detection knobs, I went back to the design drawing board. The bot recommended killfeed row detection, but it was unstable just by tracking pixel and color change. We tried using visual fingerprinting (basically, taking a slice of a killfeed region and comparing the diff between a series of frames to determine if they are a single event). Even that was unstable. Instead, we implemented a parser layer --> first layer runs detection and extraction of 'best' shots of kill feed; second layer splits the killfeed into left, middle, and right regions; then it looks for icon anchors like contoured edges; it requires left and right anchors to exist, and will reject partial animation frames. It builds parsed events by comparing the 'best' candidate frame back to the 'earliest captured' candidate frame, and uses the first indexed candidate frame as its referent. 
   1.  At least that's what it was supposed to do. Still having trouble resolving the earlier candidate frame. Tomorrow, plan to define the middle of the indexed row to better establish what a kill row is.  
10. 3.14 -- A new design paradigm. Before we were looking at lots of motion and full ROI changes with messy apportionments to intuit when changes were happening. This introduced two failure modes: 1) we were missing kills from kill parser, 2) we were duplicating kills in kill parser, which meant we were throwing out things we wanted and keeping garbage we didn't. Top of the funnel worked fine = all kills were accounted for. This suggests the problem was further down.
    1.  Use a detected kill event to kick off a sub-region detection. The first thing the subregion ID looks for is the kill arrow, because the kill arrow deterministically signposts every other important element of the UI (e.g., player icons and nameplates). The length of player names is dynamic, which makes the arrow's xy coords dynamic, so we use a coarse +-1.5x IQR from the median [25%, 75%] to predict where the arrow likely will be in a kill event window.
    2.  With the arrow's dims defined, we use offsets to identify the starting positions of every other part of the UI -- icons, nameplates -- and define the dims of those subregions too. 
    3.  This gives us the tooling to determine NEW deaths -> once an event window is triggered, we can increase the polling rate of the sampled frames in our subregions to more faithfully answer "is this kill the same as the previous kill?"
    4.  Everytime another kill event trips, the window resets and the subregions get elevated attention.
    5.  We're now seeing some regression (we dropped true kills from parse_validated) but we can implement:
        1. Bias best selection to images from raw where the kill has arrow && BOTH icons && BOTH nameplates (or some combination). 
        2. Introduce a higher image comparison threshold once a kill event starts. This means subsequent frames for a brief period must be VERY different to be counted as a new kill. This helps prune duplication.
 11. 3.15 -- Arrow detection is 100%, which allowed us to proxy detect the icons and nameplates. I realized that the better anchor is not left && right icons && nameplates, but ONLY right icon for the purposes of detecting "is this a new kill". Reason: once someone dies, they must wait at least 10s before respawning, effectively removing them from the killfeed for at least 40 seconds. So while the attacker can appear repeatedly, the victim cannot, so the victim is a better proxy for 'unique kill event'. 
     1.  We also implemented a better 'best-frame' solution: we biased it to pick candidates that had:
         1.  arrow present > both icons fully rendered > both icons mostly visible > both nameplates visible > combined icon coverage 
 12. 3.17 -- Still seeing duplications in parsed_valid. Started emitting metrics to determine where a candidate image lost in our pipeline (crops? > best selection? > burst selection? > id'd as event? > deduped? > processed by parser? invalidated by parser? > merged away as parsed event?)

