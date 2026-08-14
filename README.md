# CdScanStraightener

Command-line tool that straightens scanned CD images. Given a folder of PNG scans of discs,
it detects each disc and the rotation that makes the printed label text read upright, then
writes a rotated copy to an output folder — **rotation only**: same canvas size, same
resolution (pHYs/DPI preserved), ICC profile and color chunks (iCCP, sRGB, gAMA, cHRM)
carried over, Lanczos resampling for maximum detail retention.

## Usage

```
dotnet run --project CdScanStraightener -- -i /path/to/scans -o /path/to/straightened
```

| Option | Description |
|---|---|
| `-i, --input <dir>` | Folder with the scanned PNGs (required) |
| `-o, --output <dir>` | Output folder (required) |
| `--dry-run` | Detect and report angles without writing files |
| `-v, --verbose` | Per-file detection details |
| `--report <file.csv>` | CSV report: angle, confidence, method, disc geometry |
| `--force-angle <deg>` | Skip detection; rotate everything by this angle (CCW) |
| `--debug-dir <dir>` | Write annotated intermediates (disc overlay, polar unwrap) |
| `--min-confidence <n>` | Below this, the image is copied unrotated with a warning (default 1.5) |
| `--overwrite` | Overwrite existing output files (default: skip) |

## How it works

1. **Disc detection** — Hough circle transform on a downscaled grayscale copy; falls back to
   Otsu thresholding + largest-contour enclosing circle, then to an assumed centered disc.
2. **Angle estimation** — projection-profile method on the label annulus (hub and rim masked
   out): the edge map is rotated through candidate angles (2° coarse sweep, 0.25° refinement)
   and scored by the variance of its horizontal row sums; upright horizontal text lines give a
   peaky profile. The top few sweep peaks are kept as candidates (plus 0°, since carefully
   placed scans are common and artwork with deliberately tilted text blocks can out-score the
   design's true upright).
3. **Orientation selection** — if the `tesseract` binary is on PATH (recommended; install
   tesseract + eng traineddata), every candidate is OCRed in both 180° orientations and the
   most legible one wins; confidence is how decisively the winner out-reads the runner-up.
   Without tesseract, the best projection peak is used with a typography heuristic for the
   180° ambiguity (weaker), and confidence is the peak-to-median variance ratio.
4. **Rotation** — `warpAffine` with Lanczos4 about the **disc center** (not the image center,
   so an off-center disc stays in place), destination size = source size, uncovered corners
   filled with the median corner (scanner background) color.
5. **Metadata** — OpenCV's PNG encoder drops ancillary chunks, so pHYs, iCCP, sRGB, gAMA and
   cHRM are copied verbatim from the source file into the output.

## Notes and limitations

- Picture-only labels (no text) score low confidence and are copied unrotated with a warning;
  tune with `--min-confidence` or fix individual files with `--force-angle`.
- Data-side scans, radially-set text, and non-Latin scripts without tesseract language data
  can defeat detection — check the CSV report and `--debug-dir` overlays for outliers.
- The Linux native OpenCV runtime is referenced
  (`OpenCvSharp4.official.runtime.linux-x64`); on Windows/macOS add the matching
  `OpenCvSharp4.runtime.*` package.

## Tests

```
dotnet test
```

Covers PNG chunk roundtripping, synthetic disc detection, rotation recovery to ±0.5°,
low-confidence behavior on featureless discs, and canvas-size preservation.
