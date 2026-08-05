# Video Buddy — Pitch Deck Outline

Twelve slides. Each: headline, visual, and speaker notes. Fill [TODO]s before
presenting; keep the demo live if the room allows — a 90-second local render
is more convincing than any slide.

---

**1. Title**
- *Visual:* product name, tagline, one striking frame from the demo reel.
- *Notes:* "An autonomous AI video studio that runs on your own hardware."
  Establish in one sentence: we sell operability, not models.

**2. The gap**
- *Visual:* split screen — "models are free" vs. "operating them isn't";
  a 200-node ComfyUI graph photo.
- *Notes:* Open video models crossed the commercial quality bar. Using them
  professionally still takes a specialist and hours per shot. Cloud services
  fix usability but break privacy and price like a taxi meter.

**3. Who hurts**
- *Visual:* three personas — agency producer, in-house brand lead, music artist.
- *Notes:* Each needs volume video, can't ship client footage to a cloud
  queue, and can't justify a six-figure render specialist. [TODO: one real
  quote per persona from pilot conversations.]

**4. The product**
- *Visual:* the architecture diagram (intake → director → orchestrator →
  judge → delivery).
- *Notes:* Walk the loop: it interviews you, directs, validates before
  spending GPU, grades its own output with a vision model, and remembers
  every job. Emphasize: local-first, single consumer GPU.

**5. Demo**
- *Visual:* live or recorded — brief → storyboard → finished clip with judge
  scores; end on the Movie Builder multi-shot sequence with cloned voice.
- *Notes:* Narrate the autonomous corrections; that is the product.

**6. Why we win**
- *Visual:* moat stack — pipeline library / judge ensemble / knowledge base.
- *Notes:* Anyone can download the same weights; nobody can download our
  accumulated operational judgment. The knowledge base compounds with every
  job — the product gets better the more it's used.

**7. Market**
- *Visual:* TAM/SAM/SOM funnel. [TODO: cited figures.]
- *Notes:* Anchor on video agencies, in-house content teams, music.
  Beachhead: privacy-sensitive commercial work the clouds can't touch.

**8. Business model**
- *Visual:* pricing tiers. [TODO: finalize — license / managed service /
  pack marketplace.]
- *Notes:* Lead with the motion that matches your first customers; the pack
  marketplace (editor, mastering-tier upscale) is the expansion story.

**9. Traction**
- *Visual:* timeline of shipped milestones; demo reel stills. [TODO: pilot
  logos, usage numbers as they exist.]
- *Notes:* Today: working end-to-end system, 29 validated workflows,
  104-test suite, white paper. Be factual — this room checks.

**10. Competition**
- *Visual:* 2×2 — cloud vs. local, manual vs. autonomous.
- *Notes:* Runway/Pika/Kling: cloud-metered, privacy-limited. ComfyUI raw:
  local but specialist-only. Video Buddy: local AND autonomous — the empty
  quadrant.

**11. Risks, answered**
- *Visual:* three rows: model licensing, model dependency, hardware drift —
  each with its mitigation.
- *Notes:* Volunteer these before they're asked. Licensing review is budgeted;
  multi-model library + one-command workflow converter kills single-model
  dependency.

**12. The ask**
- *Visual:* round size, use of proceeds, 18-month milestones. [TODO.]
- *Notes:* Tie the money to the roadmap: productize packs, ship the editor
  and mastering tier, convert pilots to paid.

---

### Appendix slides (hold in reserve)
- A. Technical deep-dive (judge ensemble weights, validation gates, 16GB
  engineering profile) — pull from `docs/WHITEPAPER.md`.
- B. Model & license inventory with commercial-use status per model.
- C. Roadmap detail: special packs, editor, production mastering tier.
- D. Full workflow library list (29 entries) with live-validation status.
