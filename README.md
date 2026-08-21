# CdScanStraightener

Two command-line tools for cleaning up optical-media scans:

- **CdScanStraightener** — straightens scanned CD/DVD **discs** (rotation only, described below).
- **CdInlayStraightener** — straightens scanned **inlays/booklets**: detects the rectangle on a
  light or dark scanner background, deskews it, fixes 90°/180° orientation via OCR (losslessly,
  quarter turns never resample), and crops so **no background pixels remain** (inscribed crop
  with a checked border postcondition; `--trim` sets the inward margin, default 2 px). Output
  dimensions change; DPI and ICC metadata are preserved. Usage mirrors the disc tool:

  ```sh
  dotnet run --project CdInlayStraightener -- -i scans/ -o straightened/ [--trim 2] [--report r.csv]
  ```

  Extra options: `--force-orientation <0|90|180|270>`, `--min-confidence` (below it the inlay is
  kept deskewed but not turned), `--ocr-langs`, `--dry-run`, `--overwrite`. Undetectable scans
  are copied untouched and flagged. No AI is involved anywhere in this tool.

## CdScanStraightener (discs)

A command-line tool that automatically straightens scanned CD/DVD images.

Given a folder of PNG scans of discs, it detects each disc and the rotation that makes the
printed label read upright, then writes a rotated copy to an output folder. The correction
is **rotation only**:

- same canvas size and pixel dimensions — nothing is cropped or scaled,
- Lanczos resampling for maximum detail retention,
- resolution (DPI) preserved,
- ICC profile, color-management, EXIF and textual metadata preserved.

## Requirements

| Component | Purpose | Required |
|---|---|---|
| .NET 10 SDK | build and run | yes |
| OpenCV native library | bundled via `OpenCvSharp4.official.runtime.linux-x64` | yes (add the matching `OpenCvSharp4.runtime.*` package on Windows/macOS) |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) + `eng` traineddata on `PATH` | orientation selection by OCR legibility | strongly recommended — without it a much weaker typography heuristic is used |
| An OpenAI-compatible vision endpoint | fallback for ambiguous labels | optional |

## Building

```sh
dotnet build
dotnet test     # unit tests
```

## Usage

```sh
dotnet run --project CdScanStraightener -- -i /path/to/scans -o /path/to/straightened
```

A progress bar with ETA is shown on stderr while running; per-file results print above it.

### Options

| Option | Description |
|---|---|
| `-i, --input <dir>` | Folder with the scanned PNGs (required). |
| `-o, --output <dir>` | Output folder, created if missing (required). |
| `--dry-run` | Detect and report angles without writing any files. |
| `-v, --verbose` | Print per-file detection details (confidence, method). |
| `--report <file.csv>` | Write a CSV report (see below). |
| `--force-angle <deg>` | Skip detection; rotate every image by this angle (degrees, counterclockwise). Useful for fixing individual stragglers. |
| `--debug-dir <dir>` | Write annotated intermediates: detected disc/hub overlay with an "up" arrow, and a polar unwrap of the label. |
| `--min-confidence <n>` | Below this confidence the image is copied unrotated with a warning (default `1.5`). |
| `--overwrite` | Overwrite existing files in the output folder (default: skip them). |
| `--verify-below <n>` | Vision-verify results whose confidence is below this value (default `3.0`); pass a large number to verify every image — slow but thorough. |
| `--ocr-langs <langs>` | Tesseract language(s) for orientation OCR, e.g. `eng` or `eng+spa` (default `auto`). With a handful of packs installed `auto` uses all of them; with a full tessdata install it narrows to the languages that actually appear on optical-media labels, since joining 120+ packs makes every invocation unusably slow. |
| `--center` | Center the disc on a white square canvas (disc diameter + safe area per side); everything outside the disc becomes white. **Changes output dimensions** (DPI is kept, so physical scale is preserved). Files below `--min-confidence` are copied completely untouched — not centered either. |
| `--safe-area <px>` | White margin around the disc when using `--center` (default `25`). |

### Recommended workflow for large batches

1. `--dry-run --report angles.csv` first; skim the report.
2. Run for real. Files whose confidence stays below the threshold are copied unrotated and
   flagged — review those via the CSV (sorted by confidence) and the `--debug-dir` overlays.
3. Fix remaining stragglers one by one with `--force-angle`.

### CSV report columns

```
file, angle_deg, confidence, applied, method, disc_cx, disc_cy, disc_radius
```

`angle_deg` is the counterclockwise correction in (−180, 180]. `applied` tells whether it
was actually used or suppressed by `--min-confidence`. `method` records which stages decided
(`logo:<template>`, `projection`, `ocr`, `heuristic`, `structure` / `structure-align` /
`structure-pick` / `structure-veto`, `baseline`, `paragraph`, `textlines`, `stable`,
`openai`, plus the disc-detection mode `hough` / `contour` / `assumed-center`).

## How it works

1. **Disc detection** — on a downscaled grayscale copy (max side 1024 px, detection only;
   the final rotation is applied at full resolution): Hough circle transform, falling back
   to Otsu thresholding + largest-contour enclosing circle, then to an assumed centered
   disc. An annulus mask isolates the printable label area, excluding the hub and the rim.
2. **Logo anchoring** — tried first, because it is the only signal that recovers the full
   360° orientation outright. Rating squares (USK/PEGI/ESRB/BBFC), platform wordmarks
   (Wii, Wii U, GameCube, PlayStation, Xbox, Sega, 3DO, PC Engine), media badges (COMPACT
   disc in its several renditions, DVD, CD-i, PC CD-ROM) and publisher marks are all
   printed upright on the label. ORB keypoints are matched against a library of upright
   template crops and a RANSAC similarity transform gives the rotation directly — no OCR,
   no 180° ambiguity, and sub-degree precision. The transform's scale component *measures*
   the DPI difference rather than assuming it, so templates work across scanners and
   resolutions; each template is additionally tried at several pre-scales, including one
   upscale, because ORB's own pyramid covers a limited range. Ten RANSAC inliers decide the
   disc — genuine matches reach dozens to hundreds, false ones rarely leave single digits.
   See [Logo templates](#logo-templates) for adding your own.
3. **Angle estimation** — classic projection-profile method: the label's edge map is
   rotated through candidate angles (2° coarse sweep, 0.25° refinement) and scored by the
   variance of its horizontal row sums; upright horizontal text lines give a peaky profile.
   The top sweep peaks (≥ 8° apart) are kept as candidates, **plus 0°** — carefully placed
   scans are common, and artwork with deliberately tilted text blocks can out-score the
   design's true upright.
4. **Arc-text estimation** — circumferential (rim/arc-set) text is invisible to the
   projection sweep, so the label annulus is polar-unwrapped into a strip where arc text
   becomes horizontal, OCRed with wraparound handling, and the strongest angular cluster of
   readable words proposes the rotations that bring the arc to the top (or bottom, seal
   style) of the disc — both 180° interpretations become candidates.
5. **Orientation selection** — with `tesseract` on `PATH`, every candidate is OCRed in both
   180° orientations and the most legible wins. Before OCR the label is CLAHE-equalized
   and, for dark labels, polarity-inverted — tesseract reads dark-on-light far better, and
   silver or black discs with faint printing defeat it entirely without this. Confidence is
   how decisively the winner out-reads the best *genuinely different* orientation. Without
   tesseract, the best projection peak is used with a typography heuristic (ink-mass
   position within text-line bands) for the 180° ambiguity, and confidence is the sweep's
   peak-to-median variance ratio.
6. **Structure estimation** — logo frames, badges and boxes are printed axis-aligned on
   most designs even when the text is deliberately tilted or runs in several directions.
   Line segments (Hough), solid rectangular badges (two-level Otsu blobs whose min-area
   rect fits tightly) and outlined boxes (convex 4–8-gon contours) vote in a
   length²-weighted orientation histogram. A dominant **rectangle** is the strongest cue:
   badges are printed long-axis horizontal, pinning the upright down to a 180° ambiguity.
   Depending on how the text-based winner relates to the structure, the result is
   **corroborated** (`+structure`), **snapped** to the exact badge angle
   (`+structure-align`), **replaced** by the badge-aligned rotation whose flip OCR prefers
   consistently at two scales (`+structure-pick`), or **refused** (`+structure-veto`) —
   refusing beats confidently applying a wrong rotation. Refinement polish is drift-capped
   (≤4°) so it can only sharpen the decided answer, never re-decide it toward a
   deliberately slanted text block.

Anything still below the confidence threshold goes through the classical escalations
below, in order, each of which can settle the answer on its own:

7. **Baseline quorum** (`+baseline`) — tesseract reads tilted text nearly as well as
   straight text, so an OCR-chosen winner can sit 10°+ off on a clear-text label. A large
   quorum of wide text lines agreeing tightly on one tilt is independent evidence: it
   confirms the axis, measures the exact correction (up to 15°, versus 3° for the ordinary
   sub-degree polish), and leaves OCR only the 180° flip to vouch for.
8. **Paragraph block** (`+paragraph`) — liner-note style labels carry a large dense text
   block, and OCRing that block alone in uniform-block mode is decisively
   orientation-sensitive where whole-disc sparse OCR is a coin flip. Character-scale
   filtering drops display titles and artwork, compactness caps reject circumferential
   text rings (locally horizontal everywhere, they read plausibly at any rotation), and
   row-versus-column banding plus the measured slope of the line blobs confirm the lines
   really run horizontally before OCR is trusted. The block is read on both plausible axes
   and must clear an absolute floor and beat all three alternatives.
9. **Segmented text lines** (`+textlines`) — the last classical resort, for plain music and
   indie labels that carry no anchorable logo and no paragraph. Whole-disc OCR renders
   their few small captions a handful of pixels tall and reads nothing, so this does what a
   production OCR pipeline does: erase non-character-scale ink, close characters into
   lines, crop each line with a margin, normalize it to ~48 px tall, and recognize it alone
   in single-line mode. It derives its own axis from the line-blob orientation histogram
   (using a *round* closing kernel — a horizontal one can only ever form lines that are
   already horizontal) and excludes the rim, whose curved legal text produces line blobs at
   every angle. Only words at confidence ≥ 75 with three or more characters may vote on the
   180° flip: inverted text yields plausible-looking tokens at ordinary confidence, and by
   raw score those outvote the genuine reading.
10. **Adaptive escalation and stability** — when OCR evidence is weak, scoring is repeated
    at higher resolution (2560 px, small print often becomes decisive), and indecisive
    results get a rotation-stability probe: the disc is re-estimated pre-rotated by 37°,
    and an answer that tracks the rotation proves the estimator follows real label
    features, upgrading the confidence classically — no model call needed. A structure veto
    is final here: an estimator locked onto a slanted text block tracks the probe perfectly,
    so stability must not resurrect what structure refuted.
11. **Vision-model fallback (optional)** — labels with text running in several directions
    (radial text, opposing blocks, arc-set titles) leave OCR unable to separate the
    orientations. For those, a vision LLM can pick: the model is never asked for an angle —
    it answers a multiple-choice question over thumbnails rendered at the precise candidate
    angles. When a dominant badge has pinned the axis, the model is only ever offered the
    two rotations that keep it horizontal, and its veto-walk may not land on a
    structure-misaligned alternative. See configuration below.
12. **Rotation** — `warpAffine` with Lanczos4 about the **disc center** (not the image
    center, so an off-center disc stays in place), destination size = source size, uncovered
    corners filled with the median scanner-background color sampled from the image corners.
13. **Metadata** — OpenCV's PNG encoder drops ancillary chunks, so `pHYs` (DPI), `iCCP`
    (ICC profile), `sRGB`, `gAMA`, `cHRM`, `eXIf` (EXIF/IFD0 — scanner make and model,
    software, timestamps, the EXIF sub-IFD) and every textual chunk (`tEXt`, `zTXt`,
    `iTXt`, which may repeat) are copied from the source file into the output via a minimal
    PNG chunk parser. The one field not copied verbatim is EXIF **`Orientation`**, which is
    forced to 1 with the chunk CRC recomputed: the straightened pixels already are the
    intended orientation, so inheriting a rotated value would make a viewer that honours it
    rotate the image a second time.

Images are processed in parallel across all CPU cores (tesseract is pinned to one thread
per invocation to avoid oversubscription).

## Logo templates

`CdScanStraightener/Templates/*.png` are grayscale crops of marks that are always printed
upright. They are copied next to the executable at build time; drop a new PNG in and it is
picked up on the next build. Cropping rules, each learned from a template that failed:

- **Crop from a disc you have verified is straight.** A crop inherits any tilt in the scan
  it came from, and every disc that template then matches inherits that same error.
- **Only the reusable mark.** A game title, a "CD Multimedia" caption or a "STEREO" line
  next to the logo produces a template that matches exactly one disc and nothing else.
- **Keep a margin around it.** Tight crops starve the corner descriptors of context: one
  publisher badge went from no match at all to 43 inliers on nothing but added margin.
- **Normalize low-contrast foil.** Silver-on-iridescent marks need `-normalize` or ORB
  finds no features in them.
- **Variants are separate templates.** The same nominal logo printed as an engraved outline
  and as a solid badge does not match itself across those renditions; rating badges are the
  exception, since their frame and footer carry most of the features.

A useful shortcut: any disc the tool already orients correctly is itself a template source
for its siblings — that is how a boxed set's third disc gets anchored from the two that
already worked.


## Vision-model configuration

Settings are read from the `OpenAI` section of `appsettings.json`, looked up next to the
executable and in the working directory; environment variables in configuration syntax
(`OpenAI__ApiKey`, `OpenAI__BaseUrl`, …) override the file.

```json
{
  "OpenAI": {
    "Enabled": true,
    "BaseUrl": "https://api.openai.com/v1",
    "ApiKey": "sk-...",
    "Model": "gpt-4o-mini",
    "VerifyModel": null,
    "MaxParallelRequests": 4
  }
}
```

`VerifyModel` optionally names a different (typically stronger) model for the YES/NO
upright verification pass, while `Model` keeps handling the multiple-choice selection;
when null, `Model` is used for both.

Any OpenAI-compatible endpoint works:

- **OpenAI**: set `ApiKey`, pick a vision-capable model (e.g. `gpt-4o-mini`).
- **LM Studio** (or Ollama, vLLM, …): point `BaseUrl` at the server, e.g.
  `"http://localhost:1234/v1"`, load a vision model (e.g. `qwen/qwen3-vl-8b`) and set it
  as `Model`; no `ApiKey` needed.

The fallback only runs for images below `--min-confidence`. A connection-level failure
disables it for the rest of the run; request timeouts are treated as congestion and only
three consecutive ones disable it. At most `MaxParallelRequests` requests (default 4) run
concurrently so a local inference server is never flooded by the parallel workers. It is
best-effort: the first failed request (server down, bad key, timeout) disables it for the
rest of the run and processing continues without it. Files it decided are tagged `+openai` in the report's
`method` column for auditing.

## Debugging detection

- `--debug-dir` writes, per image, a `*.disc.png` overlay (detected disc circle, hub
  exclusion, and an arrow showing where "up" will point after correction) and a
  `*.polar.png` angular unwrap of the label.
- Setting the environment variable `CDSCAN_DEBUG=1` prints every candidate angle with its
  projection confidence and OCR legibility score to stderr.

## Known limitations

- **Picture-only labels** (no text) can't be oriented by any text-based method. If they
  carry a known mark the logo anchor still handles them; otherwise they score low
  confidence and are copied unrotated with a warning.
- **Labels whose only text is circumferential** — plain music CDs whose track list and
  legal notice both curve around the rim — leave nothing straight to read. Curved text is
  locally horizontal at every rotation, so it cannot settle the 180° flip.
- **Deliberately tilted or arc-set artwork** is inherently ambiguous — the design's
  "upright" is an artistic choice. These end up in the low-confidence set for the vision
  fallback or manual review.
- **Data-side scans** have nothing to orient by.
- **Non-Latin scripts** need the matching tesseract traineddata (`jpn` and friends ship
  with the full tessdata install); the fallback typography heuristic assumes Latin
  ascender/descender statistics.
- Rotation by arbitrary angles necessarily resamples pixels once (Lanczos); only exact
  0°/90°/180°/270° would be mathematically lossless, and straightening generally isn't.

## Tests

`dotnet test` covers PNG chunk roundtripping (including EXIF orientation normalization
and repeated textual chunks), synthetic disc detection, rotation recovery
to ±0.5°, low-confidence behavior on featureless discs, and canvas-size preservation.
