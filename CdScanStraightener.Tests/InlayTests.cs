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
        Assert.Equal(4.2, -InlayDetector.Skew(inlay.Rect), 0.3);
    }

    [Theory]
    [InlineData(255, 255, 255)]
    [InlineData(24,  24,  24)]
    public void CropContainsNoBackgroundBorderPixels(int b, int g, int r)
    {
        var bg = new Scalar(b, g, r);

        using var scan  = MakeScan(bg, -3.1);
        var       inlay = InlayDetector.Detect(scan)!;

        using var crop = Cropper.DeskewAndCrop(scan, inlay, InlayDetector.Skew(inlay.Rect), 2);

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
}
