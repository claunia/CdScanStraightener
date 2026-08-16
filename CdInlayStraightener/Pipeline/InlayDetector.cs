using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

public sealed record Inlay(RotatedRect Rect, Scalar Background);

/// <summary>
/// Finds the rectangular inlay in a scan with either a light or a dark scanner background:
/// the background reference is the median of the outer frame, the foreground is whatever
/// differs from it, and the inlay is the minimum-area rectangle of the largest blob.
/// </summary>
public static class InlayDetector
{
    /// <summary>Median color of the outer ~1% frame — the scanner background.</summary>
    public static Scalar DetectBackground(Mat src)
    {
        var t = Math.Max(4, Math.Min(src.Rows, src.Cols) / 100);

        var strips = new[]
        {
            new Rect(0, 0, src.Cols, t),
            new Rect(0, src.Rows - t, src.Cols, t),
            new Rect(0, 0, t, src.Rows),
            new Rect(src.Cols - t, 0, t, src.Rows),
        };

        var b = new List<byte>();
        var g = new List<byte>();
        var r = new List<byte>();

        foreach(var strip in strips)
        {
            using var roi = src[strip];
            // Sample sparsely — exact medians over megapixel strips are unnecessary.
            for(var y = 0; y < roi.Rows; y += 3)
                for(var x = 0; x < roi.Cols; x += 7)
                {
                    var px = roi.At<Vec3b>(y, x);
                    b.Add(px.Item0);
                    g.Add(px.Item1);
                    r.Add(px.Item2);
                }
        }

        b.Sort();
        g.Sort();
        r.Sort();

        return new Scalar(b[b.Count / 2], g[g.Count / 2], r[r.Count / 2]);
    }

    /// <summary>Binary mask of everything sufficiently different from the background color.</summary>
    public static Mat ForegroundMask(Mat src, Scalar background, double tolerance = 40)
    {
        using var blurred = new Mat();
        Cv2.GaussianBlur(src, blurred, new Size(5, 5), 0);

        using var diff = new Mat();
        Cv2.Absdiff(blurred, background, diff);

        using var diffGray = new Mat();
        Cv2.CvtColor(diff, diffGray, ColorConversionCodes.BGR2GRAY);

        var mask = new Mat();
        Cv2.Threshold(diffGray, mask, tolerance, 255, ThresholdTypes.Binary);

        // Close pin holes inside the inlay, drop dust specks on the background.
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(15, 15));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel);

        return mask;
    }

    /// <summary>Detects the inlay rectangle, or null if nothing plausibly inlay-sized is found.</summary>
    public static Inlay? Detect(Mat src)
    {
        var background = DetectBackground(src);

        using var mask = ForegroundMask(src, background);

        var points = SignificantForeground(mask);

        if(points is null) return null;

        var rect = Cv2.MinAreaRect(points);

        return new Inlay(rect, background);
    }

    /// <summary>
    /// Union of all significant foreground blobs. Inlays with large near-background (white)
    /// panels split into several blobs under color thresholding; since each scan holds
    /// exactly one inlay, every non-dust blob belongs to it and the union outlines it.
    /// Returns null when the union is implausibly small.
    /// </summary>
    public static Point[]? SignificantForeground(Mat mask)
    {
        Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External,
                         ContourApproximationModes.ApproxSimple);

        var minBlob = 0.001 * mask.Rows * mask.Cols;

        // Scanner-edge shadows appear as thin strips hugging the image border; they are
        // not part of the inlay and would stretch the union to the whole scan.
        bool IsEdgeStrip(Point[] c)
        {
            var bb = Cv2.BoundingRect(c);

            var touchesBorder = bb.Left <= 2 || bb.Top <= 2 || bb.Right >= mask.Cols - 2 ||
                                bb.Bottom >= mask.Rows - 2;

            var thin = bb.Width < 0.03 * mask.Cols || bb.Height < 0.03 * mask.Rows;

            return touchesBorder && thin;
        }

        // Dust and scanner-glass texture form sprawling speckle clouds: large bounding
        // boxes with almost no filled area. Real inlay panels are nearly solid.
        bool IsSpeckleCloud(Point[] c)
        {
            var bb = Cv2.BoundingRect(c);

            return Cv2.ContourArea(c) < 0.15 * bb.Width * bb.Height;
        }

        var kept = contours.Where(c => Cv2.ContourArea(c) >= minBlob && !IsEdgeStrip(c) && !IsSpeckleCloud(c))
                           .ToArray();

        if(Environment.GetEnvironmentVariable("CDSCAN_ARC_DUMP") is { } dump)
        {
            Cv2.ImWrite(Path.Combine(dump, "inlay-mask.png"), mask);

            foreach(var c in kept)
            {
                var bb = Cv2.BoundingRect(c);
                Console.Error.WriteLine($"    blob {bb.X},{bb.Y} {bb.Width}x{bb.Height} area={Cv2.ContourArea(c):0}");
            }
        }

        if(kept.Length == 0) return null;

        var points = kept.SelectMany(c => c).ToArray();
        var union  = Cv2.MinAreaRect(points);

        if(union.Size.Width * union.Size.Height < 0.20 * mask.Rows * mask.Cols) return null;

        return points;
    }

    /// <summary>Skew of the rectangle normalized to (−45°, 45°].</summary>
    public static double Skew(RotatedRect rect)
    {
        var a = rect.Angle % 90;

        if(a > 45) a  -= 90;
        if(a <= -45) a += 90;

        return a;
    }
}
