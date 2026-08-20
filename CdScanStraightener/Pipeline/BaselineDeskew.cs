using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Sub-degree residual correction: once the orientation is decided, the exact tilt is
/// measured from the label's own text baselines. The label is rendered at the chosen
/// angle, text characters are merged into line blobs with a wide morphological closing,
/// and the width-weighted median of the near-horizontal blob angles is the residual.
/// </summary>
public static class BaselineDeskew
{
    /// <summary>
    /// Residual tilt in degrees (CCW-positive correction), or null if too little text.
    /// <paramref name="strong"/> is set when a large quorum of wide text lines agrees
    /// tightly — evidence solid enough to correct well beyond the usual sub-degree range
    /// (an OCR-chosen winner can sit 10°+ off on a clear-text label, because tesseract
    /// reads tilted text almost as well as straight text).
    /// </summary>
    public static double? Residual(Mat gray, Disc disc, double angle, out bool strong)
    {
        strong = false;
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

        // Binarize text strokes: local adaptive threshold, both polarities merged.
        using var bin = new Mat();
        Cv2.AdaptiveThreshold(rotated, bin, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 31, 12);

        // Central annulus only: near the rim even straight design elements curve, and
        // their blobs would vote a spurious tilt.
        using var mask   = DiscDetector.AnnulusMask(gray.Size(), disc, outerFrac: 0.75);
        using var masked = new Mat();
        bin.CopyTo(masked, mask);

        // Merge characters into text-line blobs.
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(21, 3));
        using var lines  = new Mat();
        Cv2.MorphologyEx(masked, lines, MorphTypes.Close, kernel);

        Cv2.FindContours(lines, out var contours, out _, RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);

        var angles = new List<(double Angle, double Weight)>();

        foreach(var contour in contours)
        {
            var rect = Cv2.MinAreaRect(contour);
            var w    = Math.Max(rect.Size.Width, rect.Size.Height);
            var h    = Math.Min(rect.Size.Width, rect.Size.Height);

            if(h < 6 || w < disc.Radius * 0.12 || w / h < 4) continue;

            // MinAreaRect angles are ambiguous; normalize to the blob's long axis in (-45, 45].
            var a = rect.Size.Width >= rect.Size.Height ? rect.Angle : rect.Angle + 90;
            a = ((a + 180) % 180 + 180) % 180;

            if(a > 90) a -= 180;
            if(Math.Abs(a) > 20) continue; // lines belonging to a different orientation don't vote

            angles.Add((a, w));
        }

        // Quorum and agreement: a couple of blobs, or blobs that disagree, prove nothing.
        if(angles.Count < 3) return null;

        // Width-weighted median.
        var    sorted = angles.OrderBy(e => e.Angle).ToList();
        var    total  = sorted.Sum(e => e.Weight);
        var    half   = total / 2;
        double cum    = 0;
        var    median = sorted[^1].Angle;

        foreach(var e in sorted)
        {
            cum += e.Weight;

            if(cum < half) continue;

            median = e.Angle; // empirically calibrated: blob angle == needed CCW correction

            break;
        }

        // Strong quorum: many lines carrying most of the width agree tightly on one tilt.
        var agreeing = sorted.Where(e => Math.Abs(e.Angle - median) <= 1.5).ToList();
        strong = angles.Count >= 5 && agreeing.Count >= 4 && agreeing.Sum(e => e.Weight) >= total * 0.7;

        // Without a strong quorum only near-horizontal agreement counts, as before.
        if(!strong)
        {
            var near = sorted.Where(e => Math.Abs(e.Angle) <= 5).ToList();

            if(near.Count < 3) return null;

            var spread = near[^1].Angle - near[0].Angle;

            if(near.Count < 5 && spread > 2.0) return null;

            var    nearHalf = near.Sum(e => e.Weight) / 2;
            double nearCum  = 0;

            foreach(var e in near)
            {
                nearCum += e.Weight;

                if(nearCum >= nearHalf) return e.Angle;
            }

            return near[^1].Angle;
        }

        return median;
    }
}
