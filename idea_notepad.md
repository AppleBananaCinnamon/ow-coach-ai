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
6. XX
7. 


