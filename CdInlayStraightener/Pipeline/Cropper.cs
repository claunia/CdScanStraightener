using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

public static class Cropper
{
    /// <summary>
    /// Deskews by <paramref name="skew"/> about the rectangle center (Lanczos, background
    /// fill) and returns the inscribed crop with the "no background pixels" postcondition
    /// enforced: after the initial trim, the 1-px border is scanned and the crop shrinks
    /// further while any border pixel still resembles the background (capped).
    /// </summary>
    public static Mat DeskewAndCrop(Mat src, Inlay inlay, double skew, int trim)
    {
        using var rot     = Cv2.GetRotationMatrix2D(inlay.Rect.Center, skew, 1.0);
        using var rotated = new Mat();

        Cv2.WarpAffine(src, rotated, rot, src.Size(), InterpolationFlags.Lanczos4, BorderTypes.Constant,
                       inlay.Background);

        // Recompute the foreground box after deskewing.
        using var mask = InlayDetector.ForegroundMask(rotated, inlay.Background);

        var points = InlayDetector.SignificantForeground(mask);

        if(points is null) return rotated.Clone();

        var box = Cv2.BoundingRect(points);

        var crop = Shrink(box, trim, rotated.Size());

        // Postcondition: no border pixel may resemble the background. Shrink up to 12 more
        // pixels per side before giving up (heavier shrink would mean detection failed).
        for(var extra = 0; extra < 12 && BorderTouchesBackground(rotated, crop, inlay.Background); extra++)
            crop = Shrink(crop, 1, rotated.Size());

        return rotated[crop].Clone();
    }

    private static Rect Shrink(Rect r, int by, Size bounds)
    {
        var x = Math.Min(r.X + by, bounds.Width - 2);
        var y = Math.Min(r.Y + by, bounds.Height - 2);
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
