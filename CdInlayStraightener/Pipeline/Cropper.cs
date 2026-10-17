using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

public static class Cropper
{
    /// <summary>
    /// Deskews by the inlay's skew about the image center (Lanczos, background fill) — the same
    /// rotation the detector measured the box in — and returns the box trimmed inward by
    /// <paramref name="trim"/> pixels per side, so the anti-aliased edge does not survive.
    /// </summary>
    public static Mat DeskewAndCrop(Mat src, Inlay inlay, int trim)
    {
        using var rot     = Cv2.GetRotationMatrix2D(new Point2f(src.Width / 2f, src.Height / 2f), inlay.Skew, 1.0);
        using var rotated = new Mat();

        Cv2.WarpAffine(src, rotated, rot, src.Size(), InterpolationFlags.Lanczos4, BorderTypes.Constant,
                       inlay.Background);

        var crop = Shrink(inlay.Box, trim, rotated.Size());

        return rotated[crop].Clone();
    }

    private static Rect Shrink(Rect r, int by, Size bounds)
    {
        var x = Math.Clamp(r.X + by, 0, bounds.Width - 1);
        var y = Math.Clamp(r.Y + by, 0, bounds.Height - 1);
        var w = Math.Max(1, r.Width - 2 * by);
        var h = Math.Max(1, r.Height - 2 * by);

        return new Rect(x, y, Math.Min(w, bounds.Width - x), Math.Min(h, bounds.Height - y));
    }

    /// <summary>True if any pixel on the crop's 1-px border is within tolerance of the background.</summary>
    public static bool BorderTouchesBackground(Mat img, Rect crop, Scalar background, double tolerance = 40)
    {
        bool Near(Vec3b px) => Math.Abs(px.Item0 - background.Val0) < tolerance &&
                               Math.Abs(px.Item1 - background.Val1) < tolerance &&
                               Math.Abs(px.Item2 - background.Val2) < tolerance;

        for(var x = crop.Left; x < crop.Right; x++)
        {
            if(Near(img.At<Vec3b>(crop.Top, x)) || Near(img.At<Vec3b>(crop.Bottom - 1, x))) return true;
        }

        for(var y = crop.Top; y < crop.Bottom; y++)
        {
            if(Near(img.At<Vec3b>(y, crop.Left)) || Near(img.At<Vec3b>(y, crop.Right - 1))) return true;
        }

        return false;
    }
}
