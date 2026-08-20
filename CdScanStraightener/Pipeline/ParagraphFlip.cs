using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// 180° flip evidence from dense text: many labels (liner-note style music CDs above all)
/// carry a large paragraph block, and OCRing that block alone in "uniform block" mode is
/// far more decisive than whole-disc sparse OCR — a paragraph read upright produces long
/// runs of confident words, upside down it produces almost nothing. The block is found as
/// the largest dense text region after the label is rotated to the candidate axis, and the
/// flipped reading is simply the same crop rotated 180°.
/// </summary>
public static class ParagraphFlip
{
    /// <summary>
    /// Legibility of the label's largest paragraph block at <paramref name="angle"/> and at
    /// angle+180, or null when the label has no usable block. <paramref name="gray"/> should
    /// be the OCR-preprocessed grayscale.
    /// </summary>
    public static (double Upright, double Flipped)? Score(Mat gray, Disc disc, double angle, string scratch)
    {
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

        if(FindBlock(rotated, disc) is not {} block) return null;

        using var crop = rotated[block];

        var upright = Ocr(crop, scratch, flipped: false);
        var flipped = Ocr(crop, scratch, flipped: true);

        return (upright, flipped);
    }

    /// <summary>
    /// The bounding box of the largest dense text region: words merged into lines and lines
    /// into blocks morphologically, keeping regions that are large, roughly axis-aligned
    /// paragraphs (moderate ink density — artwork is denser, stray text sparser) and
    /// contain several distinct text lines.
    /// </summary>
    public static Rect? FindBlock(Mat rotated, Disc disc)
    {
        using var bin = new Mat();
        Cv2.AdaptiveThreshold(rotated, bin, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 31, 12);

        using var mask   = DiscDetector.AnnulusMask(rotated.Size(), disc);
        using var masked = new Mat();
        bin.CopyTo(masked, mask);

        // Keep only character-scale components: big display titles (often printed
        // vertically on music labels) and artwork blobs would otherwise merge into the
        // paragraph and poison its OCR.
        using var labels = new Mat();
        using var stats  = new Mat();
        using var cents  = new Mat();
        var       n      = Cv2.ConnectedComponentsWithStats(masked, labels, stats, cents);
        var       maxDim = disc.Radius * 0.06;

        for(var i = 1; i < n; i++)
        {
            var w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            var h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);

            if(w <= maxDim && h <= maxDim) continue;

            var x0 = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
            var y0 = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);

            using var region      = masked[new Rect(x0, y0, w, h)];
            using var labelRegion = labels[new Rect(x0, y0, w, h)];
            using var isThis      = new Mat();
            Cv2.Compare(labelRegion, i, isThis, CmpTypes.EQ);
            region.SetTo(Scalar.Black, isThis);
        }

        // Merge characters into lines, then lines into blocks. The kernels scale with the
        // disc so the same logic works at any resolution.
        var wordGap = Math.Max(9, (int)(disc.Radius * 0.03)) | 1;
        var lineGap = Math.Max(9, (int)(disc.Radius * 0.025)) | 1;

        using var lineKernel  = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(wordGap, 3));
        using var blockKernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, lineGap));
        using var lines       = new Mat();
        Cv2.MorphologyEx(masked, lines, MorphTypes.Close, lineKernel);
        using var blocks = new Mat();
        Cv2.MorphologyEx(lines, blocks, MorphTypes.Close, blockKernel);

        Cv2.FindContours(blocks, out var contours, out _, RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);

        Rect? best     = null;
        var   bestArea = 0.0;

        foreach(var contour in contours)
        {
            var box = Cv2.BoundingRect(contour);

            if(box.Width < disc.Radius * 0.35 || box.Height < disc.Radius * 0.2) continue;

            // Compactness: a real paragraph is a block, not a ring. Circumferential
            // annotation text is locally horizontal everywhere and OCRs plausibly at any
            // rotation, but its bounding box spans the whole label — reject it.
            if(box.Width > disc.Radius * 1.5 || box.Height > disc.Radius * 0.9) continue;

            var area = (double)box.Width * box.Height;

            if(area <= bestArea) continue;

            // Paragraph density: ink fraction of the raw binarization inside the box.
            // Artwork regions saturate; a lone title line or scattered words stay sparse.
            using var inside = masked[box];
            var       ink    = Cv2.CountNonZero(inside) / area;

            if(ink is < 0.03 or > 0.55) continue;

            // Several distinct text lines: row-projection of the line image inside the box
            // must alternate between inked and empty bands — and more strongly than the
            // column projection does, or the "lines" are actually running vertically
            // (tesseract can produce sizeable scores even on rotated text, so geometry
            // must confirm the axis before OCR is trusted).
            using var lineCrop = lines[box];

            int Bands(ReduceDimension dim)
            {
                using var sums = new Mat();
                Cv2.Reduce(lineCrop, sums, dim, ReduceTypes.Avg, MatType.CV_32FC1.Value);
                sums.GetArray(out float[] values);
                var count  = 0;
                var inBand = false;

                foreach(var v in values)
                {
                    var on = v > 32;

                    if(on && !inBand) count++;
                    inBand = on;
                }

                return count;
            }

            var rowBands = Bands(ReduceDimension.Column);
            var colBands = Bands(ReduceDimension.Row);

            if(rowBands < 4 || rowBands < colBands * 1.3) continue;

            // The line blobs themselves must lie near-horizontal: the horizontal closing
            // can manufacture horizontal-looking bands out of closely spaced vertical or
            // diagonal lines, so measure the elongated blobs' actual long-axis angles.
            Cv2.FindContours(lineCrop, out var lineBlobs, out _, RetrievalModes.External,
                             ContourApproximationModes.ApproxSimple);

            var slopes = new List<double>();

            foreach(var blob in lineBlobs)
            {
                var r = Cv2.MinAreaRect(blob);
                var w = Math.Max(r.Size.Width, r.Size.Height);
                var h = Math.Min(r.Size.Width, r.Size.Height);

                if(h < 4 || w / Math.Max(1, h) < 3) continue;

                var a = r.Size.Width >= r.Size.Height ? r.Angle : r.Angle + 90;
                a = ((a % 180) + 180) % 180;

                if(a > 90) a -= 180;
                slopes.Add(Math.Abs(a));
            }

            if(slopes.Count < 3) continue;

            slopes.Sort();

            if(slopes[slopes.Count / 2] > 15) continue;

            best     = box;
            bestArea = area;
        }

        return best;
    }

    private static double Ocr(Mat crop, string scratch, bool flipped)
    {
        using var oriented = new Mat();

        if(flipped)
            Cv2.Rotate(crop, oriented, RotateFlags.Rotate180);
        else
            crop.CopyTo(oriented);

        // Margins matter: tesseract reads poorly right up against the image edge.
        using var padded = new Mat();
        Cv2.CopyMakeBorder(oriented, padded, 20, 20, 20, 20, BorderTypes.Constant, Scalar.White);

        var tmp = Path.Combine(scratch, $"para-{Guid.NewGuid():N}.png");

        try
        {
            Cv2.ImWrite(tmp, padded);

            // psm 6: a single uniform block of text.
            return TesseractOcr.RunTsv(tmp, psm: 6) is {} words ? TesseractOcr.Score(words) : 0;
        }
        finally
        {
            if(File.Exists(tmp)) File.Delete(tmp);
        }
    }
}
