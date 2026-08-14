using CdScanStraightener.Cli;
using CdScanStraightener.Io;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public sealed record AngleResult
    (string File, double Angle, double Confidence, string Method, Point2f DiscCenter, float DiscRadius, bool Applied);

public static class StraightenPipeline
{
    private const int DetectionMaxSide = 1024;

    public static AngleResult ProcessFile(string inputPath, string outputPath, Options options)
    {
        using var src = Cv2.ImRead(inputPath, ImreadModes.Color);

        if(src.Empty()) throw new InvalidDataException($"Could not read image: {inputPath}");

        // Downscale for detection; rotation is applied at full resolution.
        var       scale = Math.Min(1.0, (double)DetectionMaxSide / Math.Max(src.Width, src.Height));
        using var gray  = new Mat();
        Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
        using var small = new Mat();

        if(scale < 1.0)
            Cv2.Resize(gray, small, default, scale, scale, InterpolationFlags.Area);
        else
            gray.CopyTo(small);

        double angle;
        double confidence;
        string method;
        var    smallDisc = DiscDetector.Detect(small);

        var fullDisc = new Disc(new Point2f((float)(smallDisc.Center.X / scale), (float)(smallDisc.Center.Y / scale)),
                                (float)(smallDisc.Radius / scale),
                                smallDisc.Method);

        if(options.ForceAngle is {} forced)
        {
            angle      = forced;
            confidence = double.PositiveInfinity;
            method     = "forced";
        }
        else
        {
            var candidates = AngleEstimator.EstimateCandidates(small, smallDisc);
            var scratch    = Path.GetTempPath();
            var scored     = new List<(double Angle, double Score)>();

            if(OcrUprightResolver.IsAvailable)
            {
                // Labels often carry deliberately tilted text blocks that win the projection
                // sweep; OCR every candidate peak in both orientations and keep the most legible.
                foreach(var candidate in candidates)
                    foreach(var orientation in new[]
                            {
                                candidate.Angle, candidate.Angle + 180
                            })
                    {
                        var score = OcrUprightResolver.OcrScore(small, smallDisc, orientation, scratch) ?? 0;

                        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                            Console.Error
                                   .WriteLine($"  {Path.GetFileName(inputPath)}: candidate {orientation:0.00}° proj-conf={candidate.Confidence:0.00} ocr={score:0}");

                        scored.Add((((orientation % 360) + 360) % 360, score));
                    }
            }

            var ranked = scored.OrderByDescending(s => s.Score).ToList();

            if(ranked.Count > 0 && ranked[0].Score > 0)
            {
                angle = ranked[0].Angle;

                // Confidence = how decisively the winning orientation out-reads the runner-up.
                var second = ranked.Count > 1 ? ranked[1].Score : 0;
                confidence = second > 0 ? ranked[0].Score / second : 10.0;
                method     = $"projection+ocr+{smallDisc.Method}";
            }
            else
            {
                angle      = UprightResolver.Resolve(small, smallDisc, candidates[0].Angle);
                confidence = candidates[0].Confidence;
                method     = $"projection+heuristic+{smallDisc.Method}";
            }
        }

        if(options.DebugDir is {} dbg) WriteDebugArtifacts(dbg, inputPath, small, smallDisc, angle);

        var applied = confidence >= options.MinConfidence;

        // Normalize to (-180, 180] so reports read naturally.
        var reportAngle = angle > 180 ? angle - 360 : angle;

        if(!options.DryRun)
        {
            using var result = applied ? Rotator.Rotate(src, fullDisc.Center, angle) : src.Clone();
            Cv2.ImWrite(outputPath, result);
            PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
        }

        return new AngleResult(Path.GetFileName(inputPath),
                               reportAngle,
                               confidence,
                               method,
                               fullDisc.Center,
                               fullDisc.Radius,
                               applied);
    }

    private static void WriteDebugArtifacts(DirectoryInfo dir, string inputPath, Mat small, Disc disc, double angle)
    {
        dir.Create();
        var stem = Path.GetFileNameWithoutExtension(inputPath);

        using var overlay = new Mat();
        Cv2.CvtColor(small, overlay, ColorConversionCodes.GRAY2BGR);
        var center = new Point((int)disc.Center.X, (int)disc.Center.Y);
        Cv2.Circle(overlay, center, (int)disc.Radius,          Scalar.Lime, 2);
        Cv2.Circle(overlay, center, (int)(disc.Radius * 0.22), Scalar.Red,  2);
        var rad = (90 + angle) * Math.PI / 180; // arrow pointing to label "up" after rotation by -angle

        var tip = new Point((int)(disc.Center.X + Math.Cos(rad) * disc.Radius * 0.8),
                            (int)(disc.Center.Y - Math.Sin(rad) * disc.Radius * 0.8));

        Cv2.ArrowedLine(overlay, center, tip, Scalar.Yellow, 2);
        Cv2.ImWrite(Path.Combine(dir.FullName, $"{stem}.disc.png"), overlay);

        using var polar = PolarUnwrapper.Unwrap(small, disc);
        Cv2.ImWrite(Path.Combine(dir.FullName, $"{stem}.polar.png"), polar);
    }
}