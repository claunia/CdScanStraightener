using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Orientation evidence from straight graphic structure: logo frames, badges, ruled
/// lines and boxes are printed axis-aligned on most disc designs, even when the text
/// blocks are deliberately tilted or run in several directions. Line segments are
/// detected on the label's edge map and their orientations (mod 90°) are histogrammed
/// with length² weighting; a sharp peak proposes the rotations that make the structure
/// axis-aligned and can corroborate an answer chosen by other estimators.
/// </summary>
public static class StructureEstimator
{
    private const double BinDeg = 0.5;

    /// <summary>
    /// Dominant structural orientation of the label in [0, 90), with a sharpness score
    /// (fraction of total segment weight inside the peak ±1.5°). Null when the label has
    /// too few long straight segments to mean anything.
    /// </summary>
    public static (double Orientation, double Sharpness)? Dominant(Mat gray, Disc disc)
    {
        using var edges = AngleEstimator.EdgeMap(gray, disc);
        using var bin   = new Mat();
        Cv2.Threshold(edges, bin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

        // Long segments only: text strokes and curved arcs must not vote. A logo frame
        // spans a good fraction of the label; text lines are broken by word gaps.
        var minLen = disc.Radius * 0.12;

        LineSegmentPoint[] segments = Cv2.HoughLinesP(bin,
                                                      1,
                                                      Math.PI / 360,
                                                      threshold: (int)(minLen * 0.8),
                                                      minLineLength: minLen,
                                                      maxLineGap: 3);

        var bins  = new double[(int)(90 / BinDeg)];
        var total = 0.0;

        void Vote(double theta, double weight)
        {
            bins[(int)((theta % 90 + 90) % 90 / BinDeg) % bins.Length] += weight;
            total                                                      += weight;
        }

        foreach(var s in segments)
        {
            double dx = s.P2.X - s.P1.X, dy = s.P2.Y - s.P1.Y;
            var    len = Math.Sqrt(dx * dx + dy * dy);

            // Image y grows downward; negate dy so angles are CCW from the x-axis.
            Vote(Math.Atan2(-dy, dx) * 180 / Math.PI, len * len);
        }

        // A tightly-filled rectangle is far stronger axis-alignment evidence than a lone
        // Hough stroke of the same length: it contributes two long and two short edges
        // that all agree — weight it accordingly. Squares vote too — a square logo frame
        // has no long axis, but mod 90 its edges pin the alignment just as well.
        foreach(var (theta, weight, _) in RectangleLongAxes(gray, disc, minAspect: 1.0)) Vote(theta, 4 * weight);

        if(total <= 0) return null;

        // Peak mass over the peak bin ±1.5° (circular mod 90).
        var reach = (int)(1.5 / BinDeg);
        var best  = (Center: 0, Mass: 0.0);

        for(var i = 0; i < bins.Length; i++)
        {
            var mass                                                          = 0.0;
            for(var j = -reach; j <= reach; j++) mass += bins[((i + j) % bins.Length + bins.Length) % bins.Length];

            if(mass > best.Mass) best = (i, mass);
        }

        // Weighted mean inside the peak window for sub-bin precision.
        double sum = 0, wsum = 0;

        for(var j = -reach; j <= reach; j++)
        {
            var idx = ((best.Center + j) % bins.Length + bins.Length) % bins.Length;
            sum  += bins[idx] * j;
            wsum += bins[idx];
        }

        if(wsum <= 0) return null;

        var orientation = (((best.Center + 0.5 + sum / wsum) * BinDeg) % 90 + 90) % 90;

        return (orientation, best.Mass / total);
    }

    /// <summary>
    /// Solid rectangular badges (e.g. a "PC CD-ROM" box) often have soft edge contrast and
    /// escape the Hough pass, but binarize into blobs whose min-area rect fits tightly.
    /// Two thresholds: global Otsu isolates light-on-dark badges, and a second Otsu computed
    /// over the sub-threshold pixels alone isolates dark-on-mid ones (a black box on a
    /// colored label is invisible to the global split). Returns the long-axis orientation
    /// in [0, 180) (CCW from the x-axis) and a length² weight per accepted rectangle.
    /// </summary>
    public static List<(double Theta, double Weight, RotatedRect Box)> RectangleLongAxes(Mat gray, Disc disc,
                                                                          double minAspect = 1.5)
    {
        var minLen = disc.Radius * 0.12;
        var rects  = new List<(double Theta, double Weight, RotatedRect Box)>();

        using var mask     = DiscDetector.AnnulusMask(gray.Size(), disc);
        using var lightBin = new Mat();
        var       t1       = Cv2.Threshold(gray, lightBin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        using var darkBin  = new Mat();
        Cv2.Threshold(gray, darkBin, OtsuBelow(gray, mask, t1), 255, ThresholdTypes.BinaryInv);

        foreach(var blobBin in new[] { lightBin, darkBin })
        {
            using var masked = new Mat();
            blobBin.CopyTo(masked, mask);

            Cv2.FindContours(masked, out var contours, out _, RetrievalModes.External,
                             ContourApproximationModes.ApproxSimple);

            foreach(var contour in contours)
            {
                var area = Cv2.ContourArea(contour);

                if(area < minLen * minLen * 0.5) continue;

                var rect = Cv2.MinAreaRect(contour);
                var w    = Math.Max(rect.Size.Width, rect.Size.Height);
                var h    = Math.Min(rect.Size.Width, rect.Size.Height);

                if(h < 4 || w < minLen || w > disc.Radius) continue;
                if(area / (w * h) < 0.85) continue; // must actually be a filled rectangle
                if(w / h < minAspect) continue;     // below this aspect the long axis means little

                // OpenCV rect angles are CW-positive in image coordinates; negate for CCW.
                var theta = rect.Size.Width >= rect.Size.Height ? -rect.Angle : -(rect.Angle + 90);
                rects.Add((((theta % 180) + 180) % 180, w * w, rect));
            }
        }

        return rects;
    }

    /// <summary>
    /// Outlined (frame-style) badges — a box drawn around text, not solid — found as closed
    /// contours that approximate a convex 4-gon with square corners. Weaker evidence than a
    /// solid rectangle (artwork can contain quadrilaterals), so callers should demand
    /// agreement between at least two boxes.
    /// </summary>
    public static List<(double Theta, double Weight, RotatedRect Box)> OutlinedBoxes(Mat gray, Disc disc, double minAspect = 1.5)
    {
        var       minLen = disc.Radius * 0.12;
        var       boxes  = new List<(double Theta, double Weight, RotatedRect Box)>();
        using var mask   = DiscDetector.AnnulusMask(gray.Size(), disc);

        foreach(var polarity in new[] { ThresholdTypes.Binary, ThresholdTypes.BinaryInv })
        {
            using var bin = new Mat();
            Cv2.AdaptiveThreshold(gray, bin, 255, AdaptiveThresholdTypes.MeanC, polarity, 51, 8);
            using var masked = new Mat();
            bin.CopyTo(masked, mask);

            Cv2.FindContours(masked, out var contours, out _, RetrievalModes.List,
                             ContourApproximationModes.ApproxSimple);

            foreach(var contour in contours)
            {
                var peri = Cv2.ArcLength(contour, true);

                if(peri < 4 * minLen * 0.8) continue;

                // Rounded corners make printed boxes approximate to 4-8 vertices, not a
                // clean 4-gon. A convex outline whose min-area rect fits this tightly can
                // only be a (rounded) rectangle: an ellipse fills just π/4 ≈ 0.79 and
                // arbitrary artwork quadrilaterals fit their min-area rect loosely.
                var poly = Cv2.ApproxPolyDP(contour, 0.02 * peri, true);

                if(poly.Length is < 4 or > 8 || !Cv2.IsContourConvex(poly)) continue;

                var rect = Cv2.MinAreaRect(poly);
                var w    = Math.Max(rect.Size.Width, rect.Size.Height);
                var h    = Math.Min(rect.Size.Width, rect.Size.Height);

                if(h < 8 || w < minLen || w > disc.Radius || w / h < minAspect) continue;
                if(Cv2.ContourArea(poly) / (w * h) < 0.87) continue;

                var theta = rect.Size.Width >= rect.Size.Height ? -rect.Angle : -(rect.Angle + 90);
                boxes.Add((((theta % 180) + 180) % 180, w * w, rect));
            }
        }

        return boxes;
    }

    /// <summary>
    /// Long-axis orientation in [0, 180) of the label's dominant rectangular badge cluster,
    /// or null when there is none or the rectangles disagree. Badges are printed with the
    /// long axis horizontal, so this pins the upright down to a 180° text ambiguity — twice
    /// as strong as the mod-90 evidence of anonymous segments. Solid rectangles are
    /// preferred; outlined boxes are consulted only when no solid one exists, and only when
    /// at least two of them agree.
    /// </summary>
    public static (double Theta, List<RotatedRect> Boxes)? RectanglePick(Mat gray, Disc disc) =>
        ClusterPick(RectangleLongAxes(gray, disc), minMembers: 1) ??
        ClusterPick(OutlinedBoxes(gray, disc), minMembers: 2);

    private static (double Theta, List<RotatedRect> Boxes)? ClusterPick(
        List<(double Theta, double Weight, RotatedRect Box)> rects, int minMembers)
    {
        if(rects.Count < minMembers || rects.Count == 0) return null;

        // Cluster mod 180 with a 3° reach and take the heaviest cluster.
        var best = (Theta: 0.0, Weight: 0.0, Members: new List<RotatedRect>());

        foreach(var (seed, _, _) in rects)
        {
            var members = rects.Where(r =>
                          {
                              var d = Math.Abs(r.Theta - seed) % 180;

                              return Math.Min(d, 180 - d) <= 3;
                          })
                                .ToList();

            var weight = members.Sum(m => m.Weight);

            if(weight > best.Weight)
                best = (members.OrderByDescending(m => m.Weight).First().Theta, weight,
                        members.Select(m => m.Box).ToList());
        }

        var total = rects.Sum(r => r.Weight);

        return best.Weight >= total * 0.7 && best.Members.Count >= minMembers
                   ? (best.Theta, best.Members)
                   : null;
    }

    /// <summary>Otsu threshold computed over the masked pixels strictly below <paramref name="upper"/>.</summary>
    private static int OtsuBelow(Mat gray, Mat mask, double upper)
    {
        gray.GetArray(out byte[] pixels);
        mask.GetArray(out byte[] inside);

        var hist = new long[256];

        for(var i = 0; i < pixels.Length; i++)
            if(inside[i] != 0 && pixels[i] < upper)
                hist[pixels[i]]++;

        long   total = hist.Sum();
        double sum   = 0;

        for(var i = 0; i < 256; i++) sum += (double)i * hist[i];

        double sumB = 0, maxVar = 0;
        long   wb   = 0;
        var    thr  = 0;

        for(var i = 0; i < 256; i++)
        {
            wb += hist[i];

            if(wb == 0) continue;

            var wf = total - wb;

            if(wf == 0) break;

            sumB += (double)i * hist[i];
            double mb = sumB / wb, mf = (sum - sumB) / wf;
            var    v  = wb * (double)wf * (mb - mf) * (mb - mf);

            if(v > maxVar)
            {
                maxVar = v;
                thr    = i;
            }
        }

        return thr;
    }

    /// <summary>
    /// The four rotations (mod 360) that make the dominant structure axis-aligned, or an
    /// empty list when there is no meaningful structure. Sharpness below
    /// <paramref name="minSharpness"/> means the segments disagree and propose nothing.
    /// </summary>
    public static List<double> Candidates(Mat gray, Disc disc, double minSharpness = 0.5)
    {
        if(Dominant(gray, disc) is not {} d || d.Sharpness < minSharpness) return [];

        // A segment at orientation θ rotated CCW by a ends at θ + a; axis-aligned means
        // θ + a ≡ 0 (mod 90), so a = -θ mod 90, in all four quadrants.
        var a = ((-d.Orientation % 90) + 90) % 90;

        return [a, a + 90, a + 180, a + 270];
    }

    /// <summary>
    /// True when <paramref name="angle"/> brings the dominant structure axis-aligned to
    /// within <paramref name="tolerance"/>° — independent corroboration of an answer
    /// chosen by text evidence. False also when there is no usable structure.
    /// </summary>
    public static bool Corroborates(Mat gray, Disc disc, double angle, double tolerance = 1.5,
                                    double minSharpness = 0.5)
    {
        if(Dominant(gray, disc) is not {} d || d.Sharpness < minSharpness) return false;

        var aligned = (d.Orientation + angle) % 90;

        return Math.Min(aligned, 90 - aligned) <= tolerance;
    }
}
