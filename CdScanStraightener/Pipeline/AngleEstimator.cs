using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public sealed record AngleEstimate(double Angle, double Confidence);

/// <summary>
/// Estimates the in-plane rotation (modulo 180°) that makes label text horizontal,
/// using the classic projection-profile method: upright horizontal text lines produce
/// a peaky horizontal projection of the edge map, i.e. maximal row-sum variance.
/// </summary>
public static class AngleEstimator
{
    public const double CoarseStep = 2.0;
    public const double FineStep   = 0.25;

    /// <summary>
    /// <paramref name="gray"/> is the (downscaled) grayscale image, <paramref name="disc"/> in its coordinates.
    /// Returns the counterclockwise rotation in [0, 180) to apply, and a confidence
    /// (peak variance over median variance across the sweep; ~1 means featureless).
    /// </summary>
    public static AngleEstimate Estimate(Mat gray, Disc disc) => EstimateCandidates(gray, disc)[0];

    /// <summary>
    /// Fine projection polish around an already-chosen angle: sweeps ±<paramref name="window"/>°
    /// at <paramref name="step"/>° and returns the best-scoring angle, preserving the chosen
    /// 180° orientation. Used as the last stage regardless of which estimator chose the angle.
    /// </summary>
    public static double RefineAround(Mat gray, Disc disc, double angle, double window = 2.0, double step = FineStep)
    {
        using var edges = EdgeMap(gray, disc);

        var best      = angle;
        var bestScore = double.MinValue;

        for(var a = angle - window; a <= angle + window; a += step)
        {
            var s = Score(edges, disc, ((a % 360) + 360) % 360);

            if(s > bestScore)
            {
                bestScore = s;
                best      = a;
            }
        }

        return ((best % 360) + 360) % 360;
    }

    /// <summary>
    /// Returns up to <paramref name="maxCandidates"/> local maxima of the sweep (best first,
    /// pairwise ≥ 8° apart mod 180), each fine-refined. Labels often contain deliberately
    /// tilted text blocks that out-score the design's true upright, so the caller should pick
    /// among candidates with a stronger signal (e.g. OCR legibility) when one is available.
    /// </summary>
    public static List<AngleEstimate> EstimateCandidates(Mat gray, Disc disc, int maxCandidates = 5)
    {
        using var edges = EdgeMap(gray, disc);

        var coarseScores                                            = new Dictionary<double, double>();
        for(double a = 0; a < 180; a += CoarseStep) coarseScores[a] = Score(edges, disc, a);

        var sorted = coarseScores.Values.Order().ToArray();
        var median = sorted[sorted.Length / 2];

        var picked = new List<double>();

        foreach(var (angle, _) in coarseScores.OrderByDescending(kv => kv.Value))
        {
            if(picked.Any(p =>
               {
                   var d = Math.Abs(p - angle) % 180;

                   return Math.Min(d, 180 - d) < 8;
               }))
                continue;

            picked.Add(angle);

            if(picked.Count == maxCandidates) break;
        }

        // Always consider "already straight": scans placed carefully on the scanner are a
        // common case, and deliberately tilted artwork can out-score the design's upright.
        if(picked.All(p => Math.Min(p, 180 - p) >= 3)) picked.Add(0);

        return picked.Select(coarse =>
                      {
                          var bestFine      = coarse;
                          var bestFineScore = coarseScores[coarse];

                          for(var a = coarse - CoarseStep; a <= coarse + CoarseStep; a += FineStep)
                          {
                              var wrapped = ((a % 180) + 180) % 180;
                              var s       = Score(edges, disc, wrapped);

                              if(s > bestFineScore)
                              {
                                  bestFineScore = s;
                                  bestFine      = wrapped;
                              }
                          }

                          var confidence = median > 1e-9 ? bestFineScore / median : 1.0;

                          return new AngleEstimate(bestFine, confidence);
                      })
                     .ToList();
    }

    /// <summary>Masked edge map emphasizing text strokes: morphological gradient of a CLAHE-equalized image.</summary>
    internal static Mat EdgeMap(Mat gray, Disc disc)
    {
        using var eq    = new Mat();
        using var clahe = Cv2.CreateCLAHE(2.0, new Size(8, 8));
        clahe.Apply(gray, eq);

        var       edges  = new Mat();
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(3, 3));
        Cv2.MorphologyEx(eq, edges, MorphTypes.Gradient, kernel);

        using var mask   = DiscDetector.AnnulusMask(gray.Size(), disc);
        var       masked = new Mat();
        edges.CopyTo(masked, mask);
        edges.Dispose();

        return masked;
    }

    /// <summary>Variance of row sums of the edge map rotated by <paramref name="angle"/> (CCW) about the disc center.</summary>
    internal static double Score(Mat edges, Disc disc, double angle)
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

        // Restrict to the central horizontal band of the disc where text rows span widest.
        var y0 = Math.Max(0, (int)(disc.Center.Y            - disc.Radius * 0.9));
        var y1 = Math.Min(rotated.Rows, (int)(disc.Center.Y + disc.Radius * 0.9));
        var x0 = Math.Max(0, (int)(disc.Center.X            - disc.Radius));
        var x1 = Math.Min(rotated.Cols, (int)(disc.Center.X + disc.Radius));

        if(y1 - y0 < 4 || x1 - x0 < 4) return 0;

        using var band    = rotated[new Rect(x0, y0, x1 - x0, y1 - y0)];
        using var rowSums = new Mat();
        Cv2.Reduce(band, rowSums, ReduceDimension.Column, ReduceTypes.Sum, MatType.CV_64FC1.Value);
        Cv2.MeanStdDev(rowSums, out _, out var stddev);

        return stddev.Val0 * stddev.Val0;
    }
}