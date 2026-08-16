using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Detects circumferential (arc-set) label text, which the projection sweep cannot see:
/// the annulus is polar-unwrapped so arc text becomes horizontal, the strip is OCRed with
/// wraparound (concatenated with itself), and each word's x position maps back to a disc
/// angle. The weighted circular mean of word angles proposes the rotation that brings the
/// text arc to the top of the disc (or, for bottom-set arcs read from the flipped strip,
/// to the bottom).
/// </summary>
public static class ArcTextEstimator
{
    private const int AngularResolution = 2048;

    public sealed record ArcCandidate(double Angle, double Score);

    /// <summary>Candidate rotations (CCW correction, [0,360)) with OCR-legibility scores.</summary>
    public static List<ArcCandidate> Candidates(Mat gray, Disc disc, string scratchDir)
    {
        var results = new List<ArcCandidate>();

        using var strip = Unwrap(gray, disc);

        // Radius increases downward in the strip, so top-set arc text (letter tops point
        // outward) needs a vertical flip to read; bottom-set arc text (seal style) needs a
        // horizontal flip, which also reverses the angle axis.
        foreach(var bottom in new[] { false, true })
        {
            using var view = new Mat();
            Cv2.Flip(strip, view, bottom ? FlipMode.Y : FlipMode.X);

            if(Estimate(view, bottom, scratchDir) is not {} c) continue;

            results.Add(c);

            // A readable arc in either view can be top-set OR bottom-set text — the two
            // interpretations differ by exactly 180°. Keep the alternative at reduced
            // weight so other evidence (whole-disc OCR, the sibling view) can outvote it.
            results.Add(new ArcCandidate((c.Angle + 180) % 360, c.Score * 0.6));
        }

        return results;
    }

    private static double CircularDistance(double a, double b)
    {
        var d = Math.Abs(a - b) % 360;

        return Math.Min(d, 360 - d);
    }

    private static double CircularMean(IReadOnlyCollection<(double Deg, double Wt)> entries)
    {
        double sx = 0, sy = 0;

        foreach(var e in entries)
        {
            var theta = e.Deg * Math.PI / 180;
            sx += e.Wt * Math.Cos(theta);
            sy += e.Wt * Math.Sin(theta);
        }

        return (Math.Atan2(sy, sx) * 180 / Math.PI + 360) % 360;
    }

    private static Mat Unwrap(Mat gray, Disc disc)
    {
        using var polar = new Mat();

        Cv2.WarpPolar(gray,
                      polar,
                      new Size((int)disc.Radius, AngularResolution),
                      disc.Center,
                      disc.Radius,
                      InterpolationFlags.Linear,
                      WarpPolarMode.Linear);

        // Rows are angle, columns radius; transpose so the angle axis runs along x and
        // crop radially to the printable annulus.
        using var strip = polar.T().ToMat();
        var r0 = (int)(strip.Rows * 0.22);
        var r1 = (int)(strip.Rows * 0.98);

        return strip[new Rect(0, r0, strip.Cols, r1 - r0)].Clone();
    }

    private static ArcCandidate? Estimate(Mat strip, bool bottom, string scratchDir)
    {
        // Duplicate horizontally so text crossing the 0°/360° seam is still readable.
        using var doubled = new Mat();
        Cv2.HConcat([strip, strip], doubled);

        var tmp = Path.Combine(scratchDir, $"arc-{Guid.NewGuid():N}.png");
        List<OcrWord>? words;

        try
        {
            Cv2.ImWrite(tmp, doubled);

            if(Environment.GetEnvironmentVariable("CDSCAN_ARC_DUMP") is { } dumpDir)
                Cv2.ImWrite(Path.Combine(dumpDir, $"strip-{(bottom ? "bottom" : "top")}.png"), doubled);

            words = TesseractOcr.RunTsv(tmp);
        }
        finally
        {
            if(File.Exists(tmp)) File.Delete(tmp);
        }

        if(words is null || words.Count == 0) return null;

        if(Environment.GetEnvironmentVariable("CDSCAN_ARC_DUMP") is not null)
            foreach(var w in words)
                Console.Error.WriteLine(
                    $"    word[{(bottom ? "B" : "T")}] '{w.Text}' x={w.X + w.W / 2.0:0} deg={(w.X + w.W / 2.0) % AngularResolution / AngularResolution * 360:0.0} conf={w.Conf:0}");

        // Word angles with weights; keep only the strongest angular cluster so stray
        // OCR hits elsewhere on the disc cannot drag the centroid.
        var entries = words.Select(w => (Deg: (w.X + w.W / 2.0) % AngularResolution / AngularResolution * 360,
                                         Wt: w.Conf * w.Text.Count(char.IsLetterOrDigit)))
                           .Where(e => e.Wt > 0)
                           .ToList();

        if(entries.Count == 0) return null;

        // Iterative trimmed circular mean: start from the global mean, then re-average over
        // the words within a symmetric window around it. Converges onto the dominant arc
        // without the truncation bias of a fixed cluster seed (an arc can span 60°+).
        double meanStripDeg = CircularMean(entries);
        double weight       = 0;

        for(var i = 0; i < 3; i++)
        {
            var kept = entries.Where(e => CircularDistance(e.Deg, meanStripDeg) <= 60).ToList();

            if(kept.Count == 0) return null;

            meanStripDeg = CircularMean(kept);
            weight       = kept.Sum(e => e.Wt);
        }

        // Coherence over ALL readable words: if text is spread around the disc (collage
        // labels) the arc interpretation is meaningless even if one window dominates.
        var coherence = weight / entries.Sum(e => e.Wt);

        if(coherence < 0.5) return null;

        // Mapping calibrated with synthetic arc-text discs: strip position → CCW correction
        // that brings the arc to the disc top (top view) or bottom (bottom view). The
        // horizontal flip of the bottom view reverses the angle axis.
        var stripDeg = bottom ? (360 - meanStripDeg) % 360 : meanStripDeg;
        var angle    = ((stripDeg - (bottom ? BottomReferenceDeg : TopReferenceDeg)) % 360 + 360) % 360;

        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
            Console.Error
                   .WriteLine($"  arc[{(bottom ? "bottom" : "top")}]: meanStrip={meanStripDeg:0.0}° -> correction {angle:0.0}° coherence={coherence:0.00} weight={weight:0}");

        return new ArcCandidate(angle, weight * coherence);
    }

    /// <summary>Strip angle (degrees) at which arc text sits when the disc is upright.</summary>
    internal static double TopReferenceDeg = 270;

    internal static double BottomReferenceDeg = 90;
}
