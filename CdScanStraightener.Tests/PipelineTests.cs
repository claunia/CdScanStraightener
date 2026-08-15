using CdScanStraightener.Pipeline;
using OpenCvSharp;

namespace CdScanStraightener.Tests;

public sealed class PipelineTests
{
    /// <summary>Synthetic CD scan: dark background, light disc, hub hole, horizontal text-like bars.</summary>
    private static Mat MakeDisc(int size = 800, double rotation = 0)
    {
        var img = new Mat(size, size, MatType.CV_8UC1, new Scalar(40));
        var c   = new Point(size / 2, size / 2);
        var r   = (int)(size * 0.45);
        Cv2.Circle(img, c, r,               new Scalar(210), -1);
        Cv2.Circle(img, c, (int)(r * 0.15), new Scalar(40),  -1);

        // Text-like horizontal bars in the label area.
        for(var i = -3; i <= 3; i++)
        {
            if(i is -1 or 0 or 1) continue; // keep the hub area clear
            var y = c.Y + i * (int)(r * 0.18);
            Cv2.Rectangle(img, new Rect(c.X - (int)(r * 0.55), y - 6, (int)(r * 1.1), 12), new Scalar(20), -1);
        }

        if(rotation != 0)
        {
            using var m       = Cv2.GetRotationMatrix2D(new Point2f(c.X, c.Y), rotation, 1.0);
            var       rotated = new Mat();

            Cv2.WarpAffine(img,
                           rotated,
                           m,
                           img.Size(),
                           InterpolationFlags.Linear,
                           BorderTypes.Constant,
                           new Scalar(40));

            img.Dispose();

            return rotated;
        }

        return img;
    }

    [Fact]
    public void DetectsSyntheticDisc()
    {
        using var img  = MakeDisc();
        var       disc = DiscDetector.Detect(img);
        Assert.Equal(400, disc.Center.X, 10.0);
        Assert.Equal(400, disc.Center.Y, 10.0);
        Assert.Equal(360, disc.Radius,   20.0);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(12.5)]
    [InlineData(-30.0)]
    [InlineData(81.0)]
    public void RecoversRotationModulo180(double applied)
    {
        using var img      = MakeDisc(rotation: applied);
        var       disc     = DiscDetector.Detect(img);
        var       estimate = AngleEstimator.Estimate(img, disc);

        // The correction should undo the applied rotation, modulo 180° (bars are symmetric).
        var diff = Math.Abs((((estimate.Angle + applied) % 180) + 180) % 180);
        diff = Math.Min(diff, 180 - diff);
        Assert.True(diff                <= 0.5, $"applied {applied}, estimated {estimate.Angle}, diff {diff}");
        Assert.True(estimate.Confidence > 1.5,  $"confidence {estimate.Confidence}");
    }

    [Fact]
    public void FeaturelessDiscHasLowConfidence()
    {
        using var img = new Mat(800, 800, MatType.CV_8UC1, new Scalar(40));
        Cv2.Circle(img, new Point(400, 400), 360, new Scalar(210), -1);
        var disc     = DiscDetector.Detect(img);
        var estimate = AngleEstimator.Estimate(img, disc);
        Assert.True(estimate.Confidence < 1.5, $"confidence {estimate.Confidence}");
    }

    [Fact]
    public void CenterOnWhiteProducesSquareCanvasWithMargin()
    {
        using var img   = MakeDisc();
        using var color = new Mat();
        Cv2.CvtColor(img, color, ColorConversionCodes.GRAY2BGR);
        var       disc     = DiscDetector.Detect(img);
        using var centered = Compositor.CenterOnWhite(color, disc.Center, disc.Radius, 25);

        var expectedSide = 2 * ((int)Math.Ceiling(disc.Radius * 1.01) + 1 + 25);
        Assert.Equal(expectedSide, centered.Width);
        Assert.Equal(expectedSide, centered.Height);

        // Corners (outside the disc circle) must be white.
        Assert.Equal(new Vec3b(255, 255, 255), centered.At<Vec3b>(2, 2));
        Assert.Equal(new Vec3b(255, 255, 255), centered.At<Vec3b>(centered.Rows - 3, centered.Cols - 3));

        // The disc's own pixels must survive at the canvas center.
        Assert.NotEqual(new Vec3b(255, 255, 255), centered.At<Vec3b>(centered.Rows / 2, centered.Cols / 4));
    }

    [Fact]
    public void RotatorPreservesCanvasSize()
    {
        using var img   = MakeDisc();
        using var color = new Mat();
        Cv2.CvtColor(img, color, ColorConversionCodes.GRAY2BGR);
        using var rotated = Rotator.Rotate(color, new Point2f(390, 410), 33.3);
        Assert.Equal(color.Size(), rotated.Size());
        Assert.Equal(color.Type(), rotated.Type());
    }
}