using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Resolves the 180° ambiguity left by projection-profile angle estimation.
/// Heuristic: a Latin text line's ink core (x-height to baseline) sits toward the
/// bottom of its full band, because the ascender/capital zone above the core is taller
/// than the descender zone below it. So within each detected text-line band the
/// edge-mass centroid sits below the band's geometric center when the text is upright.
/// </summary>
public static class UprightResolver
{
    /// <summary>Returns <paramref name="angle"/> or angle+180, whichever reads upright.</summary>
    public static double Resolve(Mat gray, Disc disc, double angle)
    {
        using var edges = AngleEstimator.EdgeMap(gray, disc);
        var       voteA = AscenderVotes(edges, disc, angle);
        var       voteB = AscenderVotes(edges, disc, angle + 180);

        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
            Console.Error.WriteLine($"upright: {angle:0.0}° votes={voteA:0.00}  {angle + 180:0.0}° votes={voteB:0.00}");

        var chosen = voteA >= voteB ? angle : angle + 180;

        return ((chosen % 360) + 360) % 360;
    }

    private static double AscenderVotes(Mat edges, Disc disc, double angle)
    {
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();

        Cv2.WarpAffine(edges,
                       rotated,
                       rot,
                       edges.Size(),
                       InterpolationFlags.Linear,
                       BorderTypes.Constant,
                       Scalar.Black);

        var y0 = Math.Max(0, (int)(disc.Center.Y            - disc.Radius * 0.9));
        var y1 = Math.Min(rotated.Rows, (int)(disc.Center.Y + disc.Radius * 0.9));
        var x0 = Math.Max(0, (int)(disc.Center.X            - disc.Radius));
        var x1 = Math.Min(rotated.Cols, (int)(disc.Center.X + disc.Radius));

        if(y1 - y0 < 8 || x1 - x0 < 8) return 0;

        using var band      = rotated[new Rect(x0, y0, x1 - x0, y1 - y0)];
        using var rowSumMat = new Mat();
        Cv2.Reduce(band, rowSumMat, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_64FC1.Value);
        var rows                                     = new double[rowSumMat.Rows];
        for(var i = 0; i < rows.Length; i++) rows[i] = rowSumMat.At<double>(i, 0);

        var mean      = rows.Average();
        var std       = Math.Sqrt(rows.Select(v => (v - mean) * (v - mean)).Average());
        var threshold = mean + 0.5 * std;

        // Group consecutive above-threshold rows into text-line bands; vote per band.
        double votes = 0;
        var    start = -1;

        for(var i = 0; i <= rows.Length; i++)
        {
            var active                    = i < rows.Length && rows[i] > threshold;
            if(active && start < 0) start = i;

            if(!active && start >= 0)
            {
                var len = i - start;

                if(len >= 3) // ignore 1-2 px noise bands
                {
                    // Centroid of edge mass within the band, relative to band center.
                    double mass   = 0, moment = 0;
                    var    center = start + (len - 1) / 2.0;

                    for(var r = start; r < i; r++)
                    {
                        mass   += rows[r];
                        moment += rows[r] * (r - center); // positive = mass below center (larger y)
                    }

                    if(mass > 0) votes += Math.Sign(moment / mass);
                }

                start = -1;
            }
        }

        return votes;
    }
}