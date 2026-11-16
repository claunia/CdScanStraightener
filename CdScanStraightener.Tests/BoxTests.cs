using CdInlayStraightener.Pipeline;
using OpenCvSharp;

namespace CdScanStraightener.Tests;

public sealed class BoxTests
{
    private const double Dpi     = 300;
    private const double PxPerMm = Dpi / 25.4;

    private static int Mm(double mm) => (int)Math.Round(mm * PxPerMm);

    /// <summary>
    /// Synthetic deskewed 300 dpi box scan on a white lid: a textured mid-grey face of 120×150 mm,
    /// optionally ringed by <paramref name="smearMm"/> of side-wall smear (stripes running along
    /// each side, darkening towards a dark silhouette line at the face), and optionally a printed
    /// dark line <paramref name="printedLineMm"/> inside the face's left edge.
    /// Returns the scan, the face rectangle and the outline the detector would report.
    /// </summary>
    private static (Mat Scan, Rect Face, Rect Outline) MakeBoxScan(double smearMm, double? printedLineMm = null)
    {
        var scan = new Mat(Mm(180), Mm(150), MatType.CV_8UC3, Scalar.All(255));
        var face = new Rect(Mm(15), Mm(15), Mm(120), Mm(150));

        // Print texture: noise of a few levels on the face, as a scan of printed card shows.
        using var noise = new Mat(face.Size, MatType.CV_8UC1);
        Cv2.Randu(noise, Scalar.All(130), Scalar.All(170));
        using var noise3 = new Mat();
        Cv2.CvtColor(noise, noise3, ColorConversionCodes.GRAY2BGR);
        noise3.CopyTo(scan[face]);

        if(printedLineMm is {} p)
            Cv2.Rectangle(scan, new Rect(face.X + Mm(p), face.Y, Mm(0.6), face.Height), Scalar.All(20), -1);

        var smear = Mm(smearMm);

        // Concentric rings outward from the face: each ring a constant tone, so the smear only
        // varies across the side, with a ramp from the lid white down towards the face and stripes.
        for(var d = smear; d >= 1; d--)
        {
            var t    = (double)d / smear;                        // 1 at the outside, 0 at the face
            var tone = 120 + 120 * t + (d / 4 % 2 == 0 ? 12 : 0); // ramp plus stripes
            var ring = new Rect(face.X - d, face.Y - d, face.Width + 2 * d, face.Height + 2 * d);

            Cv2.Rectangle(scan, ring, Scalar.All(Math.Min(250, tone)), 1);
        }

        // The silhouette: a sharp dark line where the face meets the side wall.
        if(smear > 0)
            Cv2.Rectangle(scan, new Rect(face.X - 1, face.Y - 1, face.Width + 2, face.Height + 2), Scalar.All(40), 1);

        var outline = new Rect(face.X - smear, face.Y - smear, face.Width + 2 * smear, face.Height + 2 * smear);

        // Copy the face back over so the noise is intact right up to its edge.
        noise3.CopyTo(scan[face]);

        if(printedLineMm is {} q)
            Cv2.Rectangle(scan, new Rect(face.X + Mm(q), face.Y, Mm(0.6), face.Height), Scalar.All(20), -1);

        return (scan, face, outline);
    }

    private static void AssertNear(Rect expected, Rect actual, double toleranceMm)
    {
        var tol = toleranceMm * PxPerMm;

        Assert.InRange(actual.Left, expected.Left - tol, expected.Left + tol);
        Assert.InRange(actual.Top, expected.Top - tol, expected.Top + tol);
        Assert.InRange(actual.Right, expected.Right - tol, expected.Right + tol);
        Assert.InRange(actual.Bottom, expected.Bottom - tol, expected.Bottom + tol);
    }

    [Theory]
    [InlineData(2.0)]
    [InlineData(5.0)]
    public void SideWallSmearIsCutAway(double smearMm)
    {
        var (scan, face, outline) = MakeBoxScan(smearMm);

        using(scan)
        {
            var refined = BoxFaceRefiner.Refine(scan, outline, Dpi, BoxFaceRefiner.BackgroundLuminance(scan));

            AssertNear(face, refined.Box, 0.5);
        }
    }

    [Fact]
    public void FlatInlayIsLeftAlone()
    {
        var (scan, face, outline) = MakeBoxScan(0);

        using(scan)
        {
            var refined = BoxFaceRefiner.Refine(scan, outline, Dpi, 255);

            AssertNear(face, refined.Box, 0.5);
        }
    }

    [Fact]
    public void PrintedLineNearTheEdgeIsNotTakenForTheFaceEdge()
    {
        var (scan, face, outline) = MakeBoxScan(0, printedLineMm: 6);

        using(scan)
        {
            var refined = BoxFaceRefiner.Refine(scan, outline, Dpi, 255);

            AssertNear(face, refined.Box, 0.5);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SilhouetteLogoTellsUprightFromUpsideDown(bool upsideDown)
    {
        using var template = Cv2.ImRead(Path.Combine(AppContext.BaseDirectory, "Templates", "apple.png"),
                                        ImreadModes.Grayscale);

        // A white apple on a dark box face, 15 mm tall at the logo matching scale (~12.8 px/mm).
        using var logo = new Mat();
        Cv2.Resize(template, logo, new Size(170, 190));
        Cv2.Threshold(logo, logo, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

        if(upsideDown) Cv2.Rotate(logo, logo, RotateFlags.Rotate180);

        using var scene = new Mat(1200, 1000, MatType.CV_8UC1, Scalar.All(20));
        using var inv   = new Mat();

        // Otsu leaves the apple bright on the template's grey: white on the dark scene.
        Cv2.Threshold(logo, inv, 127, 235, ThresholdTypes.Binary);
        Cv2.Max(scene[new Rect(400, 300, inv.Width, inv.Height)], inv, scene[new Rect(400, 300, inv.Width, inv.Height)]);

        Assert.Equal(upsideDown ? 180 : 0, SilhouetteAnchor.Resolve(scene, 64, 768));
    }
}
