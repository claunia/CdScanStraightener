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
    /// <summary>Residual tilt in degrees (CCW-positive correction), or null if too little text.</summary>
    public static double? Residual(Mat gray, Disc disc, double angle)
    {
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

        // Binarize text strokes: local adaptive threshold, both polarities merged.
        using var bin = new Mat();
        Cv2.AdaptiveThreshold(rotated, bin, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.BinaryInv, 31, 12);

        using var mask   = DiscDetector.AnnulusMask(gray.Size(), disc);
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
            if(Math.Abs(a) > 5) continue; // only near-horizontal lines vote on residual tilt

            angles.Add((a, w));
        }

        if(angles.Count < 2) return null;

        // Width-weighted median.
        var sorted = angles.OrderBy(e => e.Angle).ToList();
        var half   = sorted.Sum(e => e.Weight) / 2;
        double cum = 0;

        foreach(var e in sorted)
        {
            cum += e.Weight;

            if(cum >= half) return e.Angle; // empirically calibrated: blob angle == needed CCW correction
        }

        return sorted[^1].Angle;
    }
}
