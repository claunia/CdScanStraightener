using CdScanStraightener.Cli;
using CdScanStraightener.Io;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public sealed record AngleResult
    (string File, double Angle, double Confidence, string Method, Point2f DiscCenter, float DiscRadius, bool Applied);

public static class StraightenPipeline
{
    private const int DetectionMaxSide = 1024;
    private const int OcrMaxSide       = 1536;

    /// <summary>
    /// Contrast preparation for tesseract: CLAHE equalization, plus polarity inversion for
    /// dark labels — tesseract reads dark-on-light far better than light-on-dark, and
    /// silver/black discs with faint printing defeat it entirely without this.
    /// </summary>
    private static void PreprocessForOcr(Mat gray, Disc disc)
    {
        using var mask = DiscDetector.AnnulusMask(gray.Size(), disc);
        var       mean = Cv2.Mean(gray, mask).Val0;

        if(mean < 128) Cv2.BitwiseNot(gray, gray);

        using var clahe = Cv2.CreateCLAHE(2.0, new Size(8, 8));
        clahe.Apply(gray, gray);
    }

    private static double AngularDistance(double a, double b)
    {
        var d = Math.Abs(a - b) % 360;

        return Math.Min(d, 360 - d);
    }

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
            var candidates = AngleEstimator.EstimateCandidates(small, smallDisc, maxCandidates: 5);
            var scratch    = Path.GetTempPath();
            var scored     = new List<(double Angle, double Score)>();

            // OCR needs more resolution than the projection sweep: small label text is
            // illegible at the detection scale, exactly where discrimination is needed.
            var       ocrScale = Math.Min(1.0, (double)OcrMaxSide / Math.Max(src.Width, src.Height));
            using var ocrGray  = new Mat();

            if(ocrScale < 1.0)
                Cv2.Resize(gray, ocrGray, default, ocrScale, ocrScale, InterpolationFlags.Area);
            else
                gray.CopyTo(ocrGray);

            var ocrDisc = new Disc(new Point2f((float)(smallDisc.Center.X / scale * ocrScale),
                                               (float)(smallDisc.Center.Y / scale * ocrScale)),
                                   (float)(smallDisc.Radius / scale * ocrScale),
                                   smallDisc.Method);

            PreprocessForOcr(ocrGray, ocrDisc);

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
                        var score = OcrUprightResolver.OcrScore(ocrGray, ocrDisc, orientation, scratch) ?? 0;

                        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                            Console.Error
                                   .WriteLine($"  {Path.GetFileName(inputPath)}: candidate {orientation:0.00}° proj-conf={candidate.Confidence:0.00} ocr={score:0}");

                        scored.Add((((orientation % 360) + 360) % 360, score));
                    }

                // Circumferential text is invisible to the projection sweep and unreadable
                // by whole-disc OCR; the polar-unwrap estimator proposes and scores it.
                foreach(var arc in ArcTextEstimator.Candidates(ocrGray, ocrDisc, scratch))
                {
                    if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                        Console.Error
                               .WriteLine($"  {Path.GetFileName(inputPath)}: arc candidate {arc.Angle:0.00}° score={arc.Score:0}");

                    scored.Add((arc.Angle, arc.Score));
                }
            }

            var ranked = scored.OrderByDescending(s => s.Score).ToList();

            if(ranked.Count > 0 && ranked[0].Score > 0)
            {
                angle = ranked[0].Angle;

                // Confidence = how decisively the winning orientation out-reads the best
                // genuinely different orientation (nearby angles read almost as well as the
                // winner by construction and must not deflate the ratio).
                var second = ranked.Skip(1)
                                   .Where(s => AngularDistance(s.Angle, ranked[0].Angle) > 20)
                                   .Select(s => s.Score)
                                   .DefaultIfEmpty(0)
                                   .Max();

                confidence = second > 0 ? ranked[0].Score / second : 10.0;
                method     = $"projection+ocr+{smallDisc.Method}";
            }
            else
            {
                angle      = UprightResolver.Resolve(small, smallDisc, candidates[0].Angle);
                confidence = candidates[0].Confidence;
                method     = $"projection+heuristic+{smallDisc.Method}";
            }

            // When OCR could not separate the orientations, ask a vision model to pick
            // among the candidate thumbnails (multiple choice — never a free-form angle).
            if(confidence < options.MinConfidence && options.OpenAi.IsUsable)
            {
                using var smallColor = new Mat();

                if(scale < 1.0)
                    Cv2.Resize(src, smallColor, default, scale, scale, InterpolationFlags.Area);
                else
                    src.CopyTo(smallColor);

                var orientations = scored.Count > 0
                                       ? ranked.Take(6).Select(s => s.Angle).ToList()
                                       : candidates.SelectMany(c => new[]
                                                    {
                                                        c.Angle, c.Angle + 180
                                                    })
                                                   .Select(a => ((a % 360) + 360) % 360)
                                                   .ToList();

                if(OpenAiOrientationResolver.Resolve(smallColor, smallDisc, orientations, options.OpenAi) is {} pick)
                {
                    angle      = pick;
                    confidence = options.MinConfidence; // the model's choice is applied
                    method     += "+openai";
                }
            }
        }

        if(options.DebugDir is {} dbg) WriteDebugArtifacts(dbg, inputPath, small, smallDisc, angle);

        var applied = confidence >= options.MinConfidence;

        // Normalize to (-180, 180] so reports read naturally.
        var reportAngle = angle > 180 ? angle - 360 : angle;

        if(!options.DryRun)
        {
            using var rotated = applied ? Rotator.Rotate(src, fullDisc.Center, angle) : src.Clone();

            // Unresolved files are copied completely untouched — no centering either, so
            // nothing destructive (the white fill discards background) happens to a file
            // whose disc geometry we are not confident about.
            using var result = applied && options.Center
                                   ? Compositor.CenterOnWhite(rotated, fullDisc.Center, fullDisc.Radius,
                                                              options.SafeArea)
                                   : rotated.Clone();

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