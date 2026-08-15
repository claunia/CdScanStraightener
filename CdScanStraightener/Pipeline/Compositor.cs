using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public static class Compositor
{
    /// <summary>
    /// Extracts the disc and centers it on a white square canvas of exactly
    /// disc diameter + 2 × <paramref name="safeArea"/> px per side. Everything outside the
    /// disc circle becomes white. Note this changes the output dimensions (DPI is kept, so
    /// physical scale is preserved).
    /// </summary>
    public static Mat CenterOnWhite(Mat src, Point2f center, float radius, int safeArea)
    {
        // A little slack so an under-estimated radius does not shave the disc edge.
        var r    = (int)Math.Ceiling(radius * 1.01) + 1;
        var side = 2 * (r + safeArea);
        var dst  = new Mat(side, side, src.Type(), Scalar.White);

        using var mask = Mat.Zeros(src.Size(), MatType.CV_8UC1).ToMat();
        var cx = (int)Math.Round(center.X);
        var cy = (int)Math.Round(center.Y);
        Cv2.Circle(mask, cx, cy, r, Scalar.White, -1, LineTypes.AntiAlias);

        // Intersect the disc's bounding box with the source (the disc may touch the scan edge).
        var srcRect = new Rect(cx - r, cy - r, 2 * r, 2 * r);
        var clipped = srcRect.Intersect(new Rect(0, 0, src.Width, src.Height));

        if (clipped.Width <= 0 || clipped.Height <= 0) return dst;

        var dstRect = new Rect(safeArea + (clipped.X - srcRect.X),
                               safeArea + (clipped.Y - srcRect.Y),
                               clipped.Width,
                               clipped.Height);

        using var srcRoi  = src[clipped];
        using var maskRoi = mask[clipped];
        using var dstRoi  = dst[dstRect];
        srcRoi.CopyTo(dstRoi, maskRoi);

        return dst;
    }
}
