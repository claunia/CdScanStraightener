using CdInlayStraightener.Pipeline;
using OpenCvSharp;

namespace CdScanStraightener.Tests;

public sealed class InlayTests
{
    /// <summary>Synthetic scan: a colored card with dark bars, tilted, on a solid background.</summary>
    private static Mat MakeScan(Scalar background, double tilt)
    {
        var scan = new Mat(1000, 1400, MatType.CV_8UC3, background);

        using var card = new Mat(500, 800, MatType.CV_8UC3, new Scalar(160, 216, 232));

        for(var i = 0; i < 4; i++)
            Cv2.Rectangle(card, new Rect(80, 80 + i * 100, 640, 24), new Scalar(30, 30, 30), -1);

        // Paste the card rotated by `tilt` about the scan center.
        using var big = new Mat(1000, 1400, MatType.CV_8UC3, background);
        card.CopyTo(big[new Rect(300, 250, 800, 500)]);
        using var m = Cv2.GetRotationMatrix2D(new Point2f(700, 500), tilt, 1.0);
        Cv2.WarpAffine(big, scan, m, scan.Size(), InterpolationFlags.Linear, BorderTypes.Constant, background);

        return scan;
    }

    [Theory]
    [InlineData(255, 255, 255)]
    [InlineData(24,  24,  24)]
    public void DetectsInlayOnAnyBackground(int b, int g, int r)
    {
        using var scan  = MakeScan(new Scalar(b, g, r), 4.2);
        var       inlay = InlayDetector.Detect(scan);

        Assert.NotNull(inlay);
        Assert.Equal(4.2, -inlay.Skew, 0.3);
    }

    [Theory]
    [InlineData(255, 255, 255)]
    [InlineData(24,  24,  24)]
    public void CropContainsNoBackgroundBorderPixels(int b, int g, int r)
    {
        var bg = new Scalar(b, g, r);

        using var scan  = MakeScan(bg, -3.1);
        var       inlay = InlayDetector.Detect(scan)!;

        using var crop = Cropper.DeskewAndCrop(scan, inlay, 2);

        Assert.False(Cropper.BorderTouchesBackground(crop, new Rect(0, 0, crop.Width, crop.Height), bg));

        // Dimensions near the card size (800×500) minus trims.
        Assert.InRange(crop.Width,  770, 800);
        Assert.InRange(crop.Height, 470, 500);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void QuarterRotationIsLosslessAndInvertible(int q)
    {
        using var scan = MakeScan(Scalar.White, 0);

        using var turned = OrientationResolver.Rotate(scan, q);
        using var back   = OrientationResolver.Rotate(turned, 4 - q);

        Assert.Equal(scan.Size(), back.Size());

        using var diff = new Mat();
        Cv2.Absdiff(scan, back, diff);
        Assert.Equal(0, Cv2.Sum(diff).Val0 + Cv2.Sum(diff).Val1 + Cv2.Sum(diff).Val2);
    }

    [Fact]
    public void FeaturelessScanIsRejected()
    {
        using var scan = new Mat(1000, 1400, MatType.CV_8UC3, Scalar.White);
        Assert.Null(InlayDetector.Detect(scan));
    }

    /// <summary>
    /// Synthetic 300 dpi Wii cover scan: 2156×3224 px paper against the scanner's edge on a white
    /// lid, spine logo and spine triangle where the layout puts them, tilted and turned a quarter.
    /// </summary>
    private static Mat MakeWiiScan(double tilt)
    {
        const int cx = 2350, cy = 1540;
        var       lid = new Scalar(251, 251, 251);

        using var canvas = new Mat(3240, 2550, MatType.CV_8UC3, lid);
        Cv2.Rectangle(canvas, new Rect(cx - 2156, cy - 1534, 2156, 3224), new Scalar(244, 244, 244), -1);

        using var logo = Cv2.ImRead(Path.Combine(AppContext.BaseDirectory, "Templates", "wii-spine.png"),
                                    ImreadModes.Color);

        logo.CopyTo(canvas[new Rect(cx - 2075, cy + 30, logo.Width, logo.Height)]);

        Cv2.FillConvexPoly(canvas, [new Point(cx, cy), new Point(cx - 160, cy), new Point(cx, cy + 165)],
                           new Scalar(200, 150, 230));

        using var m      = Cv2.GetRotationMatrix2D(new Point2f(1275, 1620), tilt, 1.0);
        using var tilted = new Mat();
        Cv2.WarpAffine(canvas, tilted, m, canvas.Size(), InterpolationFlags.Linear, BorderTypes.Constant, lid);

        var scan = new Mat();
        Cv2.Rotate(tilted, scan, RotateFlags.Rotate90Counterclockwise);

        return scan;
    }

    [Fact]
    public void WiiCoverIsFramedByItsLandmarks()
    {
        using var scan  = MakeWiiScan(0.6);
        using var cover = WiiCover.Detect(scan, 300);

        Assert.NotNull(cover);
        Assert.Equal(0.6, Math.Abs(cover.Skew), 0.15);

        // Upright and whole: 273 mm wide, 183 mm tall, less what the tilt pushed off the scan.
        Assert.InRange(cover.Image.Width, 3180, 3230);
        Assert.InRange(cover.Image.Height, 2120, 2160);
    }

    [Fact]
    public void PlainInlayIsNotTakenForAWiiCover()
    {
        using var scan = MakeScan(Scalar.White, 2);
        Assert.Null(WiiCover.Detect(scan, 300));
    }
}
