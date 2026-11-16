using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

/// <summary>
/// Upright-or-upside-down from a plain silhouette logo — the Apple apple — which keypoint
/// matching cannot see: a smooth outline has no corners to describe. The logo's largest blob
/// (the apple's body) is found as a stable region of either polarity; around it, where the
/// whole logo would be if upright or if upside down, the scan is split into body-like and
/// background-like pixels and compared with the template's shape. The leaf above the body
/// and the bite on its right tell the two apart.
/// </summary>
public static class SilhouetteAnchor
{
    private const int Side = 64;

    private sealed record Template(string Name, Mat Upright, Mat Flipped, double BodyAspect, double BodyFill,
                                   double X0, double Y0, double X1, double Y1);

    /// <summary>Silhouette logos among the logo templates.</summary>
    private static readonly string[] Names = ["apple"];

    private static readonly Lazy<List<Template>> Templates = new(Load);

    /// <summary>
    /// 0 when a silhouette logo stands upright in <paramref name="gray"/>, 180 when it is upside
    /// down, null when none is found unambiguously. Logos are expected between
    /// <paramref name="minPx"/> and <paramref name="maxPx"/> tall.
    /// </summary>
    public static int? Resolve(Mat gray, int minPx, int maxPx)
    {
        if(Templates.Value.Count == 0) return null;

        using var mser = MSER.Create(5, minPx * minPx / 4, maxPx * maxPx);
        mser.DetectRegions(gray, out var regions, out var boxes);

        double bestScore = 0;
        int?   best      = null;

        for(var r = 0; r < regions.Length; r++)
        {
            var b = boxes[r];

            if(b.Height < minPx * 0.7 || b.Height > maxPx) continue;

            foreach(var t in Templates.Value)
            {
                var aspect = (double)b.Width / b.Height;
                var fill   = (double)regions[r].Length / (b.Width * b.Height);

                if(Math.Abs(aspect - t.BodyAspect) > 0.15 * t.BodyAspect || Math.Abs(fill - t.BodyFill) > 0.12) continue;

                var up   = Score(gray, regions[r], b, t, false);
                var down = Score(gray, regions[r], b, t, true);

                // Decisive: a close fit one way and clearly worse the other.
                var (score, angle, other) = up >= down ? (up, 0, down) : (down, 180, up);

                if(score < 0.8 || score - other < 0.1 || score <= bestScore) continue;

                bestScore = score;
                best      = angle;
            }
        }

        return best;
    }

    /// <summary>IoU of the scene around a body blob with the template, upright or upside down.</summary>
    private static double Score(Mat gray, Point[] body, Rect b, Template t, bool flipped)
    {
        // Where the whole logo lies relative to the body's box; mirrored both ways when flipped.
        double x0 = t.X0, y0 = t.Y0, x1 = t.X1, y1 = t.Y1;

        if(flipped) (x0, y0, x1, y1) = (1 - x1, 1 - y1, 1 - x0, 1 - y0);

        var frame = new Rect((int)Math.Round(b.X + x0 * b.Width), (int)Math.Round(b.Y + y0 * b.Height),
                             (int)Math.Round((x1 - x0) * b.Width), (int)Math.Round((y1 - y0) * b.Height));

        if(frame.X < 0 || frame.Y < 0 || frame.Right > gray.Cols || frame.Bottom > gray.Rows || frame.Width < 8 ||
           frame.Height < 8)
            return 0;

        // Body tone from the blob, background tone from the rest of the frame.
        var bodyTone = body.Average(p => (double)gray.At<byte>(p.Y, p.X));

        using var region = gray[frame].Clone();
        using var inBody = new Mat(frame.Size, MatType.CV_8U, Scalar.All(0));

        foreach(var p in body)
            if(frame.Contains(p))
                inBody.Set(p.Y - frame.Y, p.X - frame.X, (byte)255);

        using var outside = new Mat();
        Cv2.BitwiseNot(inBody, outside);

        var backTone = Cv2.Mean(region, outside).Val0;

        if(Math.Abs(bodyTone - backTone) < 25) return 0;

        // Body-like pixels: nearer the body tone than the background tone.
        using var mask = new Mat();
        Cv2.Threshold(region, mask, (bodyTone + backTone) / 2, 255,
                      bodyTone > backTone ? ThresholdTypes.Binary : ThresholdTypes.BinaryInv);

        using var scaled = new Mat();
        Cv2.Resize(mask, scaled, new Size(Side, Side), 0, 0, InterpolationFlags.Area);
        Cv2.Threshold(scaled, scaled, 127, 255, ThresholdTypes.Binary);

        return Iou(scaled, flipped ? t.Flipped : t.Upright);
    }

    private static double Iou(Mat a, Mat b)
    {
        using var and = new Mat();
        using var or  = new Mat();
        Cv2.BitwiseAnd(a, b, and);
        Cv2.BitwiseOr(a, b, or);

        var union = Cv2.CountNonZero(or);

        return union == 0 ? 0 : (double)Cv2.CountNonZero(and) / union;
    }

    private static List<Template> Load()
    {
        var list = new List<Template>();

        var dir = new[]
                  {
                      Path.Combine(AppContext.BaseDirectory, LogoAnchor.TemplateFolder),
                      Path.Combine(Environment.CurrentDirectory, LogoAnchor.TemplateFolder)
                  }.FirstOrDefault(Directory.Exists);

        if(dir is null) return list;

        foreach(var name in Names)
        {
            var file = Path.Combine(dir, name + ".png");

            if(!File.Exists(file)) continue;

            using var img  = Cv2.ImRead(file, ImreadModes.Grayscale);
            using var mask = new Mat();

            // The logo is the minority tone: Otsu, then make it the foreground.
            Cv2.Threshold(img, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

            if(Cv2.CountNonZero(mask) > mask.Total() / 2) Cv2.BitwiseNot(mask, mask);

            using var labels    = new Mat();
            using var stats     = new Mat();
            using var centroids = new Mat();
            var       count     = Cv2.ConnectedComponentsWithStats(mask, labels, stats, centroids);

            if(count < 2) continue;

            var body = Enumerable.Range(1, count - 1).MaxBy(i => stats.At<int>(i, (int)ConnectedComponentsTypes.Area));
            var bb = new Rect(stats.At<int>(body, 0), stats.At<int>(body, 1), stats.At<int>(body, 2),
                              stats.At<int>(body, 3));
            var bodyArea = stats.At<int>(body, (int)ConnectedComponentsTypes.Area);

            using var points = new Mat();
            Cv2.FindNonZero(mask, points);
            var full = Cv2.BoundingRect(points);

            using var cropped = mask[full].Clone();
            var       upright = new Mat();
            Cv2.Resize(cropped, upright, new Size(Side, Side), 0, 0, InterpolationFlags.Area);
            Cv2.Threshold(upright, upright, 127, 255, ThresholdTypes.Binary);

            var flipped = new Mat();
            Cv2.Rotate(upright, flipped, RotateFlags.Rotate180);

            list.Add(new Template(name, upright, flipped, (double)bb.Width / bb.Height,
                                  (double)bodyArea / (bb.Width * bb.Height), (double)(full.X - bb.X) / bb.Width,
                                  (double)(full.Y - bb.Y) / bb.Height, (double)(full.Right - bb.X) / bb.Width,
                                  (double)(full.Bottom - bb.Y) / bb.Height));
        }

        return list;
    }
}
