using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public sealed record Disc(Point2f Center, float Radius, string Method);

/// <summary>Finds the CD in a scan: HoughCircles first, contour fallback, centered assumption last.</summary>
public static class DiscDetector
{
    public static Disc Detect(Mat gray)
    {
        var minSide = Math.Min(gray.Width, gray.Height);

        using(var blurred = new Mat())
        {
            Cv2.GaussianBlur(gray, blurred, new Size(9, 9), 2);

            var circles = Cv2.HoughCircles(blurred,
                                           HoughModes.Gradient,
                                           dp: 1.5,
                                           minDist: minSide,
                                           param1: 120,
                                           param2: 60,
                                           minRadius: (int)(minSide * 0.30),
                                           maxRadius: (int)(minSide * 0.52));

            if(circles.Length > 0)
            {
                var c = circles[0];

                return new Disc(new Point2f(c.Center.X, c.Center.Y), c.Radius, "hough");
            }
        }

        // Fallback: Otsu threshold, largest contour, min enclosing circle.
        using(var bin = new Mat())
        {
            Cv2.Threshold(gray, bin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

            // The disc may be darker or lighter than the background; try both polarities.
            foreach(var invert in new[]
                    {
                        false, true
                    })
            {
                using var work = invert ? (Mat)(Scalar.All(255) - bin) : bin.Clone();

                Cv2.FindContours(work,
                                 out var contours,
                                 out _,
                                 RetrievalModes.External,
                                 ContourApproximationModes.ApproxSimple);

                var best = contours.Select(c => (Contour: c, Area: Cv2.ContourArea(c)))
                                   .OrderByDescending(t => t.Area)
                                   .FirstOrDefault();

                if(best.Contour is null) continue;
                Cv2.MinEnclosingCircle(best.Contour, out var center, out var radius);

                // Sanity: circle must be disc-sized and mostly filled (circular blob, not scanner lid).
                var circularity = best.Area / (Math.PI * radius * radius);

                if(radius >= minSide * 0.25 && radius <= minSide * 0.55 && circularity > 0.7)
                    return new Disc(center, radius, "contour");
            }
        }

        return new Disc(new Point2f(gray.Width / 2f, gray.Height / 2f), minSide * 0.48f, "assumed-center");
    }

    /// <summary>Mask keeping only the printable label annulus (excludes hub area and outer rim).</summary>
    public static Mat AnnulusMask(Size size, Disc disc, double innerFrac = 0.22, double outerFrac = 0.95)
    {
        var mask   = Mat.Zeros(size, MatType.CV_8UC1).ToMat();
        var center = new Point((int)disc.Center.X, (int)disc.Center.Y);
        Cv2.Circle(mask, center, (int)(disc.Radius * outerFrac), Scalar.White, -1);
        Cv2.Circle(mask, center, (int)(disc.Radius * innerFrac), Scalar.Black, -1);

        return mask;
    }
}