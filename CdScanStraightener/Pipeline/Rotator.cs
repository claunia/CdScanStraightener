using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public static class Rotator
{
    /// <summary>
    /// Rotates <paramref name="src"/> by <paramref name="angle"/> degrees (CCW) about the disc
    /// center — not the image center, so an off-center disc stays in place — keeping the exact
    /// canvas size, with Lanczos resampling. Uncovered corners are filled with the background
    /// color sampled from the image corners.
    /// </summary>
    public static Mat Rotate(Mat src, Point2f center, double angle)
    {
        using var rot = Cv2.GetRotationMatrix2D(center, angle, 1.0);
        var       dst = new Mat();

        Cv2.WarpAffine(src,
                       dst,
                       rot,
                       src.Size(),
                       InterpolationFlags.Lanczos4,
                       BorderTypes.Constant,
                       CornerMedianColor(src));

        return dst;
    }

    private static Scalar CornerMedianColor(Mat src)
    {
        var s = Math.Max(2, Math.Min(src.Rows, src.Cols) / 100);

        var corners = new[]
        {
            new Rect(0, 0, s, s), new Rect(src.Cols - s, 0, s, s), new Rect(0, src.Rows - s, s, s),
            new Rect(src.Cols                       - s, src.Rows                       - s, s, s),
        };

        var means = corners.Select(r =>
                            {
                                using var roi = src[r];

                                return Cv2.Mean(roi);
                            })
                           .ToArray();

        return new Scalar(Median(means.Select(m => m.Val0)),
                          Median(means.Select(m => m.Val1)),
                          Median(means.Select(m => m.Val2)),
                          Median(means.Select(m => m.Val3)));

        static double Median(IEnumerable<double> vals)
        {
            var a = vals.Order().ToArray();

            return a[a.Length / 2];
        }
    }
}