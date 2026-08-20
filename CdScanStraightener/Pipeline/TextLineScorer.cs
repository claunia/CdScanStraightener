using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Orientation evidence from properly segmented text lines. Whole-disc OCR renders a
/// small caption a few pixels tall and reads nothing, which is why sparse-mode scoring is
/// indecisive on plain music and indie labels — the discs that carry no anchorable logo.
/// This scorer instead does what a production OCR pipeline does: isolate character-scale
/// ink, merge it into lines, crop each line with a margin, normalize it to a standard
/// height, and recognize it on its own in single-line mode.
///
/// It also derives its own axis from the detected lines rather than trusting a candidate
/// angle, so a projection sweep that locked onto artwork cannot mislead it.
/// </summary>
public static class TextLineScorer
{
    /// <summary>Target height in pixels for a normalized line crop — tesseract's comfort zone.</summary>
    private const double LineHeight = 48.0;

    public sealed record Reading(double Angle, double Score, int Lines);

    /// <summary>
    /// The best orientation by line-level legibility, or null when the label carries too
    /// little text to judge. Candidate axes come from the dominant text-line direction;
    /// both 180° interpretations of each are read and the winner is returned with the
    /// runner-up so the caller can judge decisiveness.
    /// </summary>
    public static (Reading Best, Reading Runner)? Evaluate(Mat gray, Disc disc, string scratch)
    {
        var axes = DominantAxes(gray, disc);

        if(axes.Count == 0) return null;

        var readings = new List<Reading>();

        foreach(var axis in axes)
            foreach(var flip in new[] { 0.0, 180.0 })
            {
                var angle = ((axis + flip) % 360 + 360) % 360;
                var (score, lines) = Score(gray, disc, angle, scratch);
                readings.Add(new Reading(angle, score, lines));
            }

        readings = readings.OrderByDescending(r => r.Score).ToList();

        return readings[0].Score <= 0 ? null : (readings[0], readings.Count > 1 ? readings[1] : new Reading(0, 0, 0));
    }

    /// <summary>
    /// Rotations that would bring the label's dominant text direction horizontal, best
    /// first. Derived from the orientation histogram of elongated text-line blobs found on
    /// the unrotated label, so it does not depend on any earlier estimate.
    /// </summary>
    private static List<double> DominantAxes(Mat gray, Disc disc)
    {
        using var lines = LineMap(gray, disc, isotropic: true);

        Cv2.FindContours(lines, out var contours, out _, RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);

        // Histogram line-blob directions (mod 180) weighted by length².
        const double binDeg = 2.0;
        var          bins   = new double[(int)(180 / binDeg)];

        foreach(var contour in contours)
        {
            var rect = Cv2.MinAreaRect(contour);
            var w    = Math.Max(rect.Size.Width, rect.Size.Height);
            var h    = Math.Min(rect.Size.Width, rect.Size.Height);

            if(h < 5 || w < disc.Radius * 0.08 || w / h < 2.5) continue;

            var theta = rect.Size.Width >= rect.Size.Height ? -rect.Angle : -(rect.Angle + 90);
            theta = ((theta % 180) + 180) % 180;
            bins[(int)(theta / binDeg) % bins.Length] += w * w;
        }

        if(bins.Sum() <= 0) return [];

        // Peaks, each at least 12° from a stronger one; at most three axes are worth OCR.
        var peaks = new List<double>();

        foreach(var (index, _) in bins.Select((v, i) => (i, v)).OrderByDescending(p => p.v))
        {
            if(bins[index] <= 0) break;

            var theta = (index + 0.5) * binDeg;

            if(peaks.Any(p =>
               {
                   var d = Math.Abs(p - theta) % 180;

                   return Math.Min(d, 180 - d) < 12;
               }))
                continue;

            peaks.Add(theta);

            if(peaks.Count == 3) break;
        }

        // Rotating by -theta brings that direction horizontal.
        return peaks.Select(t => ((-t % 360) + 360) % 360).ToList();
    }

    /// <summary>
    /// Text-line map: character-scale ink only (artwork and display lettering would
    /// otherwise merge into one mass that no closing can separate), morphologically closed
    /// along the horizontal so characters join into lines.
    /// </summary>
    /// <param name="isotropic">
    /// Merge characters with a round kernel instead of a horizontal one. A horizontal
    /// kernel only ever forms lines that are already horizontal, so it cannot be used to
    /// discover which way the text runs; a round one merges characters whatever the
    /// direction, letting the blob's own long axis reveal it.
    /// </param>
    private static Mat LineMap(Mat gray, Disc disc, double rotation = 0, bool isotropic = false)
    {
        using var rotated = new Mat();

        if(Math.Abs(rotation) < 1e-9)
            gray.CopyTo(rotated);
        else
        {
            using var rot = Cv2.GetRotationMatrix2D(disc.Center, rotation, 1.0);
            Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Cubic, BorderTypes.Constant,
                           Scalar.Black);
        }

        using var bin = new Mat();
        Cv2.AdaptiveThreshold(rotated, bin, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 31, 12);

        // Exclude the rim: circumferential legal text curves, so it never forms a straight
        // readable line, but at an arbitrary rotation it does produce line blobs at every
        // angle — enough to swamp the axis histogram of the label's real text.
        using var mask   = DiscDetector.AnnulusMask(rotated.Size(), disc, outerFrac: 0.80);
        using var masked = new Mat();
        bin.CopyTo(masked, mask);

        KeepCharacterScale(masked, disc);

        var gap = Math.Max(9, (int)(disc.Radius * 0.025)) | 1;

        using var kernel = isotropic
                               ? Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(gap, gap))
                               : Cv2.GetStructuringElement(MorphShapes.Rect, new Size(gap, 3));

        var lines = new Mat();
        Cv2.MorphologyEx(masked, lines, MorphTypes.Close, kernel);

        return lines;
    }

    /// <summary>Erases connected components too large to be characters — artwork, rules, big display type.</summary>
    private static void KeepCharacterScale(Mat binary, Disc disc)
    {
        using var labels = new Mat();
        using var stats  = new Mat();
        using var cents  = new Mat();
        var       count  = Cv2.ConnectedComponentsWithStats(binary, labels, stats, cents);
        var       maxDim = disc.Radius * 0.11;

        for(var i = 1; i < count; i++)
        {
            var w = stats.At<int>(i, (int)ConnectedComponentsTypes.Width);
            var h = stats.At<int>(i, (int)ConnectedComponentsTypes.Height);

            if(w <= maxDim && h <= maxDim) continue;

            var       x = stats.At<int>(i, (int)ConnectedComponentsTypes.Left);
            var       y = stats.At<int>(i, (int)ConnectedComponentsTypes.Top);
            using var region      = binary[new Rect(x, y, w, h)];
            using var labelRegion = labels[new Rect(x, y, w, h)];
            using var isThis      = new Mat();
            Cv2.Compare(labelRegion, i, isThis, CmpTypes.EQ);
            region.SetTo(Scalar.Black, isThis);
        }
    }

    /// <summary>
    /// Total legibility of the label's text lines after rotating by <paramref name="angle"/>:
    /// each near-horizontal line is cropped with a margin, scaled so its height lands in
    /// tesseract's comfort zone, and recognized in single-line mode.
    /// </summary>
    private static (double Score, int Lines) Score(Mat gray, Disc disc, double angle, string scratch)
    {
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Cubic, BorderTypes.Constant, Scalar.Black);

        using var lines = LineMap(rotated, disc);

        Cv2.FindContours(lines, out var contours, out _, RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);

        var total = 0.0;
        var used  = 0;

        foreach(var contour in contours)
        {
            var rect = Cv2.MinAreaRect(contour);
            var w    = Math.Max(rect.Size.Width, rect.Size.Height);
            var h    = Math.Min(rect.Size.Width, rect.Size.Height);

            if(h < 5 || w < disc.Radius * 0.08 || w / h < 2.5) continue;

            var theta = rect.Size.Width >= rect.Size.Height ? rect.Angle : rect.Angle + 90;
            theta = ((theta % 180) + 180) % 180;

            if(theta > 90) theta -= 180;
            if(Math.Abs(theta) > 8) continue; // this rotation did not make the line horizontal

            var box    = Cv2.BoundingRect(contour);
            var margin = (int)(box.Height * 0.35) + 3;
            var x0     = Math.Max(0, box.X                    - margin);
            var y0     = Math.Max(0, box.Y                    - margin);
            var x1     = Math.Min(rotated.Cols, box.X + box.Width  + margin);
            var y1     = Math.Min(rotated.Rows, box.Y + box.Height + margin);

            if(x1 - x0 < 20 || y1 - y0 < 8) continue;

            using var crop = rotated[new Rect(x0, y0, x1 - x0, y1 - y0)];
            using var norm = new Mat();
            var       up   = Math.Clamp(LineHeight / (y1 - y0), 1.0, 8.0);
            Cv2.Resize(crop, norm, default, up, up, InterpolationFlags.Cubic);
            using var padded = new Mat();
            Cv2.CopyMakeBorder(norm, padded, 25, 25, 25, 25, BorderTypes.Replicate);

            var tmp = Path.Combine(scratch, $"line-{Guid.NewGuid():N}.png");

            try
            {
                Cv2.ImWrite(tmp, padded);

                if(TesseractOcr.RunTsv(tmp, psm: 7) is not {} words) continue;

                // Upside-down text still yields tokens at ordinary confidence — enough
                // garbage to outscore the genuine reading. Only words the recognizer is
                // really sure of, and long enough to be words, may vote on the flip.
                var confident = words.Where(w => w.Conf >= 75 && w.Text.Count(char.IsLetterOrDigit) >= 3).ToList();

                if(confident.Count == 0) continue;

                total += TesseractOcr.Score(confident);
                used++;
            }
            finally
            {
                if(File.Exists(tmp)) File.Delete(tmp);
            }
        }

        return (total, used);
    }
}
