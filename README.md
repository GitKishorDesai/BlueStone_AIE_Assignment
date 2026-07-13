# BlueStone Matching Jewellery Set Recommender

Given a jewellery design, recommends complementary products (e.g. earrings
for a pendant, a ring for a necklace) to form a matching set — combining
visual similarity (CLIP embeddings), structured metadata, and business
rules.

Live demo: https://bluestone-aie-assignment.onrender.com  
(Note: Application is deployed on Free Tier, so there might be some Start-Up Time)

---

## Architecture

<img src='./architecture.png'>

Offline: CSV of design IDs → fetch product API → normalize → LLM gender
fallback → download images → CLIP embeddings → Chroma vector index.

Runtime: anchor product → Chroma similarity search (pre-filtered by
is_kids) → hard filters (category, zone, gender) → weighted scoring →
rank + diversify by category → top-K with explanations → served via
FastAPI to the frontend.

---
## How It Works

1. **One-Time Steps**: fetch product data → normalize fields →
   fill gaps with an LLM where needed → generate image embeddings → index
   everything in a vector DB (Chroma).
2. **Runtime app** (deployed): given an anchor product, retrieve visually
   similar candidates from Chroma → apply hard filters → score on weighted
   criteria → rank → return top-K with explanations. The homepage also
   shows 3 randomly-generated example matching sets on load (via a
   `/discover` endpoint), so the system is demonstrable without needing
   to search first.
3. **AI evaluation (on-demand)**: after a matching set is generated, the
user can trigger a second, independent AI evaluation of that set — a
multimodal call (Gemini 3.5 Flash) that looks at the actual product
images plus structured attributes and gives a buyer-facing verdict,
dimension scores, and a "weakest link" callout. This is a qualitative
second opinion on the algorithm's own output, not part of the
recommendation logic itself. Gated behind a button on search results
(to avoid unnecessary LLM cost on every query); auto-run on the 3
featured homepage examples using a cheaper text-only version, since
those are bounded in number and meant to showcase the feature by
default.

See [Architecture](#architecture) below for the full flow.

---

## Repository structure

```
config.py                       # settings: paths, API URLs, category rules
sampler.py                      # offline: CSV -> sample of design IDs
fetcher.py                      # offline: calls product API, caches responses
inspect_data.py                 # offline: reports field/category completeness
normalize.py                    # offline: raw data -> clean per-product record
enrich_gender.py                # offline: LLM fallback for ambiguous gender
download_images.py              # offline: downloads product images
generate_embeddings.py          # offline: CLIP embedding per image
build_vector_index.py           # offline: builds the Chroma vector index
data_loader.py                  # runtime: loads data, connects to Chroma
matching.py                     # runtime: retrieval, filters, scoring, ranking
api.py                          # runtime: FastAPI app (deployed)
static/index.html               # runtime: frontend UI
data/
    /chroma_db                  # Vector Database
    products.db                 # sqlite3 Database
```

Only `config.py`, `data_loader.py`, `matching.py`, `api.py`,
`static/index.html`, and `data/products.db` + `data/chroma_db/` are
needed to run the deployed app. Everything else is a one-time data-build
step, kept in the repo for reproducibility.

---

## Setup

**1. Clone the repo:**
```bash
git clone <your-repo-url>
cd bluestone_assignment
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**3. Run the app.** The repo already includes the precomputed dataset
(`data/normalized`, `data/embeddings`, `data/chroma_db`), so no data-build
step is needed — just start the server:
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000` in your browser.

---

### Optional: rebuild the dataset from scratch

Only needed if you want to regenerate the data (e.g. with a different
sample size), not required to just run the app.

```bash
python sampler.py active_design_ids.csv --sample-size 100 --id-column design_id
python fetcher.py
python normalize.py
python enrich_gender.py        # needs GROQ_API_KEY in a .env file
python download_images.py
python generate_embeddings.py
python build_vector_index.py
```

### Optional: quick CLI testing (bypasses the UI)

```bash
python matching.py --design-id 433 --top-k 5
python matching.py                          # random anchor if no ID given
```

---

## Data exploration findings

Development and initial design decisions were based on a 96-product
sample; the full pipeline was later re-run against the complete dataset
of ~8,185 design IDs, of which 437 (≈5.3%) failed to resolve via the
Product Details API (confirmed as delisted/unavailable, both via the API
and manually on BlueStone's own site) — yielding **7,748 valid products**
in the final catalog.

Field-completeness ratios held consistent between the 96-sample and the
full 7,748-product run (e.g. missing gender: 13.5% in-sample vs. 12.7% at
full scale; missing collection: 76% vs. 78.2%), validating that the
original small sample was representative. Key findings:

## Matching logic

**Hard Filters** (exclude candidates entirely):
- Same Category excluded (category names canonicalized, e.g. "Rings" and
  "PreSet Solitaire Rings" treated as one category)
- Same "Zone Group" excluded by default — necklace / pendant / mangalsutra
   are all worn in Neck, so for a given Necklace, Pendant and Mangalsutra will not be recommended in the Matching Set(toggle: `include_same_zone`)
- Gender must be compatible (unisex matches anything; Men/Women must match)
- Adult vs. Kids must match (filtered directly in the Chroma query)

**Soft-Scored Criteria** (weighted, summed into a final match score):

| Criterion | Weight | Why |
|---|---|---|
| Visual similarity (CLIP) | 0.30 | Captures design language/motif — nothing else covers this |
| Metal + finish match | 0.25 | Mismatched metal color is the most visually jarring mismatch |
| Gemstone compatibility | 0.20 | Diamond/colored/plain consistency matters for "set" feel |
| Price-tier closeness | 0.10 | A real business constraint |
| Collection match | 0.10 | Strong signal, but present on only ~24% of products |
| Occasion/tag overlap | 0.05 | Weakest, sparsest signal |

Weights are **manually reasoned, not learned** — no co-purchase/click data
was available to fit them against. Missing data (e.g. no collection on
either side) contributes 0, never a penalty.

Explanations are template-based (deterministic, from which criteria scored
highest), not LLM-generated — fast, free, reproducible.

---

## Where AI is Used

- **CLIP Embeddings** (`ViT-B-32`) — one per product image, compared via
  cosine similarity at query time. This is the visual/design-language
  signal, and the reason a vector DB (Chroma) is used for retrieval.
- **Groq (llama-3.1-8b-instant)** — text-only, used only to fill in gender when
  category + tags + design name give no clear signal. After deriving
  gender from structured fields first, ~ 1000/7748 records needed this,
  and non-ring categories skip the LLM entirely (defaulted to "women" per
  observed catalog skew) — so the LLM call is a narrow fallback, not a
  bulk step.
  - **Gemini (`gemini-3.5-flash`), multimodal** — used only for the
  on-demand "Evaluate this set with AI" feature. Given the anchor
  product's image + attributes and each recommended item's image +
  attributes, it produces a buyer-facing verdict (strong/acceptable/weak
  match), scores four dimensions (metal & finish, gemstone & color,
  style & occasion, overall quality), and explicitly names the single
  weakest-fitting item in the set, if any. Images are passed directly by
  public URL (no download/base64 step needed) via the Gemini Interactions
  API's multi-image input support.


  **Prompt used:**
```
  You are classifying the target gender for a jewellery product based on its metadata.

  Product details:
  - Category: {category_type}
  - Sub-category: {item_category_name}
  - Design name: {design_name}
  - Existing tag keywords: {tag_count} tags present

  Classify the target gender as one of: "women", "men", "unisex".
  If the product name/category gives no clear signal, default to "unisex".

  Respond with ONLY valid JSON: {"gender": "women", "confidence": "high", "reasoning": "brief reason"}
```

```
You are an expert jewellery stylist helping a customer decide whether a
recommended set of jewellery pieces is worth buying together as a matching set. 

You are shown the actual product images below, in this order: first the anchor
piece the customer is already interested in, then each recommended piece.  

Anchor product attributes:
{anchor_summary}  
Recommended set attributes:
{recs_summary}  

Look at the images and reason using both what you see and the attributes above.
Do not invent details not visible in the images or listed above.  

Write for the customer directly, in a warm, confident, concise tone —
2-3 sentences maximum, no jewellery-analyst jargon, no bullet lists in the summary text.  
Also identify the single weakest-fitting item in the set, if any (there may be
none, in which case say so) — name it explicitly and explain briefly why a
customer might want to treat it as optional rather than essential.  

Score four dimensions 1-10 based on what you see in the images plus the
attributes: metal_finish, gemstone_color, style_occasion, overall_quality.  

Give an overall_verdict: one of "strong match", "acceptable match", "weak match".

Respond with ONLY valid JSON, no other text, in exactly this schema:  
  {"buyer_summary": "...",
  "weakest_item": "...",
  "weakest_item_reason": "...",
  "metal_finish": {"score": 8, "reasoning": "..."},
  "gemstone_color": {...},
  "style_occasion": {...},
  "overall_quality": {...},
  "overall_verdict": "strong match"}
```

## Evaluation Feature — Design Notes

The AI-evaluation feature is intentionally split into two tiers:

- **Featured homepage examples** (auto-run, 3 per page load): text-only,
  Groq — cheap and fast, since this runs unconditionally on every visit.
- **User-triggered search evaluation** (button click): multimodal, Gemini
  — richer and slower, since it only runs when a person deliberately asks
  for it, and 6 images per call (anchor + up to 5 recommendations) is
  meaningfully heavier than a text-only call.

This evaluation layer is independent of the matching algorithm itself —
it's a second, qualitative opinion on an already-generated set, not part
of how the set is chosen. It can flag issues the deterministic scoring
missed (or vice versa disagree with it), which is the point: two
different reasoning approaches checking the same output.

## Scaling from 96 to 7,748 products — issues found and fixed

Running the full pipeline at full catalog scale surfaced two real issues
that the 96-sample was too small to expose:

- **Same-category candidates weren't excluded early enough.** At small
  scale, Chroma's top-N visual-similarity results often included enough
  cross-category items by chance. At full scale, visually-similar results
  for popular categories (e.g. Rings) were dominated entirely by
  same-category items, causing some queries to return zero
  recommendations after hard filtering. Fixed by pushing category
  exclusion directly into the Chroma query itself (via a
  `canonical_category` metadata field + `$ne` filter), rather than only
  filtering in Python after retrieval.
- **New category types not covered by manually-curated config.** The full
  catalog revealed category labels absent from the 96-sample (e.g.
  standalone "Chains" and "Mangalsutra" as distinct from "Mangalsutra
  Chains"), which weren't included in the neck-zone exclusion group,
  and a "Nose Rings" category that incorrectly canonicalized to the
  generic "ring" bucket due to a substring match. Both fixed once
  identified via a full category-distribution scan.

This is a concrete example of why the two-tier sample-then-scale approach
was useful — these issues would have been much harder to catch and debug
against the full ~8k catalog directly.

---

## Key Assumptions

- Category-pairing "strength" is not tiered (e.g. necklace+earrings isn't
  scored higher than ring+bracelet) — noted as a future improvement.
- Non-ring categories with ambiguous gender default to "women", based on
  observed catalog skew, not asserted as fact.
- Gemstone matching compares kind only (diamond/colored/solitaire/plain),
  not carat or clarity.
- Scoring weights are hand-reasoned, not learned from data.

---

## Known Limitations / Future Improvements

- Category-pairing strength could be tiered or learned from co-purchase data.
- Scoring weights are unvalidated against real outcomes — a learned ranking
  model would be the natural next step in production.
- Design/motif matching is currently implicit (via embedding similarity);
  adding zero-shot CLIP motif labels (e.g. "floral", "geometric") would make
  explanations more concrete.
- No automated recommendation-quality metric exists — no ground truth was
  available; outputs were reviewed manually throughout development.
- The AI evaluation feature's two tiers (text-only vs. multimodal) use
  different prompts/schemas by design; a future version could unify them
  behind one schema with an optional "images available" flag.
---

