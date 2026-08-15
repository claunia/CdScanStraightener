# CdScanStraightener

A command-line tool that automatically straightens scanned CD/DVD images.

Given a folder of PNG scans of discs, it detects each disc and the rotation that makes the
printed label read upright, then writes a rotated copy to an output folder. The correction
is **rotation only**:

- same canvas size and pixel dimensions — nothing is cropped or scaled,
- Lanczos resampling for maximum detail retention,
- resolution (DPI) preserved,
- ICC profile and color-management chunks preserved.

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
| `--ocr-langs <langs>` | Tesseract language(s) for orientation OCR, e.g. `eng` or `eng+spa` (default `auto` = all installed packs). |
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
(`projection`, `ocr`, `heuristic`, `openai`, and the disc-detection mode `hough` /
`contour` / `assumed-center`).

## How it works

1. **Disc detection** — on a downscaled grayscale copy (max side 1024 px, detection only;
   the final rotation is applied at full resolution): Hough circle transform, falling back
   to Otsu thresholding + largest-contour enclosing circle, then to an assumed centered
   disc. An annulus mask isolates the printable label area, excluding the hub and the rim.
2. **Angle estimation** — classic projection-profile method: the label's edge map is
   rotated through candidate angles (2° coarse sweep, 0.25° refinement) and scored by the
   variance of its horizontal row sums; upright horizontal text lines give a peaky profile.
   The top sweep peaks (≥ 8° apart) are kept as candidates, **plus 0°** — carefully placed
   scans are common, and artwork with deliberately tilted text blocks can out-score the
   design's true upright.
3. **Arc-text estimation** — circumferential (rim/arc-set) text is invisible to the
   projection sweep, so the label annulus is polar-unwrapped into a strip where arc text
   becomes horizontal, OCRed with wraparound handling, and the strongest angular cluster of
   readable words proposes the rotations that bring the arc to the top (or bottom, seal
   style) of the disc — both 180° interpretations become candidates.
4. **Orientation selection** — with `tesseract` on `PATH`, every candidate is OCRed in both
   180° orientations and the most legible wins. Before OCR the label is CLAHE-equalized
   and, for dark labels, polarity-inverted — tesseract reads dark-on-light far better, and
   silver or black discs with faint printing defeat it entirely without this. All installed language packs are used by
   default (`--ocr-langs` overrides) — install the packs matching your discs' languages. Confidence is how decisively the winner
   out-reads the runner-up. Without tesseract, the best projection peak is used with a
   typography heuristic (ink-mass position within text-line bands) for the 180° ambiguity,
   and confidence is the sweep's peak-to-median variance ratio.
5. **Vision-model fallback (optional)** — labels with text running in several directions
   (radial text, opposing blocks, arc-set titles) leave OCR unable to separate the
   orientations. For those, a vision LLM can pick: the model is never asked for an angle —
   it answers a multiple-choice question over thumbnails rendered at the precise candidate
   angles. See configuration below.
6. **Rotation** — `warpAffine` with Lanczos4 about the **disc center** (not the image
   center, so an off-center disc stays in place), destination size = source size, uncovered
   corners filled with the median scanner-background color sampled from the image corners.
7. **Metadata** — OpenCV's PNG encoder drops ancillary chunks, so `pHYs` (DPI), `iCCP`
   (ICC profile), `sRGB`, `gAMA` and `cHRM` are copied verbatim from the source file into
   the output via a minimal PNG chunk parser.

Images are processed in parallel across all CPU cores (tesseract is pinned to one thread
per invocation to avoid oversubscription).

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

The fallback only runs for images below `--min-confidence`. At most `MaxParallelRequests` requests (default 4) run
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

- **Picture-only labels** (no text) can't be oriented by any text-based method; they score
  low confidence and are copied unrotated with a warning.
- **Deliberately tilted or arc-set artwork** is inherently ambiguous — the design's
  "upright" is an artistic choice. These end up in the low-confidence set for the vision
  fallback or manual review.
- **Data-side scans** have nothing to orient by.
- **Non-Latin scripts** need the matching tesseract traineddata; the fallback typography
  heuristic assumes Latin ascender/descender statistics.
- Rotation by arbitrary angles necessarily resamples pixels once (Lanczos); only exact
  0°/90°/180°/270° would be mathematically lossless, and straightening generally isn't.

## Tests

`dotnet test` covers PNG chunk roundtripping, synthetic disc detection, rotation recovery
to ±0.5°, low-confidence behavior on featureless discs, and canvas-size preservation.
