using CdScanStraightener.Cli;
using CdScanStraightener.Common;
using CdScanStraightener.Io;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

public sealed record AngleResult
    (string File, double Angle, double Confidence, string Method, Point2f DiscCenter, float DiscRadius, bool Applied);

public static class StraightenPipeline
{
    private const int DetectionMaxSide = 1024;
    private const int OcrMaxSide       = 1536;
    private const int OcrEscalatedSide = 2560;

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

    /// <summary>
    /// OCR-legibility sweep ±8° in 2° steps around an angle. Tesseract tolerates small
    /// skews, so scores plateau near the optimum; taking the raw argmax random-walks on
    /// that plateau's noise. Instead the smallest offset whose score is within 5% of the
    /// maximum wins, so the angle only moves when OCR clearly improves.
    /// </summary>
    private static double RefineWithOcr(Mat gray, Disc disc, double angle, string scratch)
    {
        var scores = new Dictionary<int, double>();

        for(var off = -8; off <= 8; off += 2) scores[off] = OcrUprightResolver.OcrScore(gray, disc, ((angle + off) % 360 + 360) % 360, scratch) ?? 0;

        var max = scores.Values.Max();

        if(max <= 0) return ((angle % 360) + 360) % 360;

        var chosen = scores.Where(kv => kv.Value >= max * 0.95)
                           .OrderBy(kv => Math.Abs(kv.Key))
                           .First()
                           .Key;

        return ((angle + chosen) % 360 + 360) % 360;
    }

    /// <summary>
    /// Second-opinion orientation estimate on a pre-rotated copy of the working images.
    /// Used for the rotation-stability confidence check: if the estimate tracks a known
    /// synthetic rotation, the estimator is locked onto real label features.
    /// </summary>
    private static double? BestOrientation(Mat small, Disc smallDisc, Mat ocrGray, Disc ocrDisc, string scratch)
    {
        var candidates = AngleEstimator.EstimateCandidates(small, smallDisc, maxCandidates: 5);
        var scored     = new List<(double Angle, double Score)>();

        foreach(var candidate in candidates)
            foreach(var orientation in new[]
                    {
                        candidate.Angle, candidate.Angle + 180
                    })
                scored.Add((((orientation % 360) + 360) % 360,
                            OcrUprightResolver.OcrScore(ocrGray, ocrDisc, orientation, scratch) ?? 0));

        foreach(var arc in ArcTextEstimator.Candidates(ocrGray, ocrDisc, scratch)) scored.Add((arc.Angle, arc.Score));

        var best = scored.OrderByDescending(sc => sc.Score).FirstOrDefault();

        return best.Score > 0 ? best.Angle : null;
    }

    private static Mat Rotated(Mat src, Point2f center, double angle)
    {
        using var m   = Cv2.GetRotationMatrix2D(center, angle, 1.0);
        var       dst = new Mat();
        Cv2.WarpAffine(src, dst, m, src.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

        return dst;
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

            // Known-logo anchoring first: rating squares, platform wordmarks and media
            // badges are printed upright, and one confident keypoint match pins the full
            // 360° orientation with sub-degree precision — stronger than any text
            // heuristic, immune to the 180° ambiguity. Run on the raw gray (before OCR
            // preprocessing mutates it).
            if(LogoAnchor.Resolve(ocrGray, minInliers: 10) is {} logo)
            {
                if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                    Console.Error
                           .WriteLine($"  {Path.GetFileName(inputPath)}: logo anchor {logo.Template} {logo.Angle:0.00}° inliers={logo.Inliers} scale={logo.Scale:0.00}");

                return Finish(inputPath, outputPath, options, src, fullDisc, small, smallDisc, logo.Angle,
                              Math.Max(5.0, options.MinConfidence), $"logo:{logo.Template}+{smallDisc.Method}");
            }

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

                // Straight graphic structure (logo frames, badges, ruled boxes) is printed
                // axis-aligned on most designs even when the text is deliberately tilted or
                // multi-directional; a sharp segment-orientation peak proposes four more
                // rotations for OCR to choose among.
                foreach(var structural in StructureEstimator.Candidates(small, smallDisc))
                {
                    if(scored.Any(sc => AngularDistance(sc.Angle, structural) < 3)) continue;

                    var score = OcrUprightResolver.OcrScore(ocrGray, ocrDisc, structural, scratch) ?? 0;

                    if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                        Console.Error
                               .WriteLine($"  {Path.GetFileName(inputPath)}: structure candidate {structural:0.00}° ocr={score:0}");

                    scored.Add((((structural % 360) + 360) % 360, score));
                }
            }

            var ranked = scored.OrderByDescending(s => s.Score).ToList();

            // Weak evidence usually means small print that is illegible at the standard OCR
            // size. Escalating resolution costs seconds and often turns an unreadable label
            // into a decisive one — far cheaper than falling back to a vision model.
            if(OcrUprightResolver.IsAvailable && (ranked.Count == 0 || ranked[0].Score < 2000))
            {
                var escScale = Math.Min(1.0, (double)OcrEscalatedSide / Math.Max(src.Width, src.Height));

                if(escScale > ocrScale)
                {
                    using var escGray = new Mat();

                    if(escScale < 1.0)
                        Cv2.Resize(gray, escGray, default, escScale, escScale, InterpolationFlags.Area);
                    else
                        gray.CopyTo(escGray);

                    var escDisc = new Disc(new Point2f((float)(smallDisc.Center.X / scale * escScale),
                                                       (float)(smallDisc.Center.Y / scale * escScale)),
                                           (float)(smallDisc.Radius / scale * escScale),
                                           smallDisc.Method);

                    PreprocessForOcr(escGray, escDisc);

                    var rescored = new List<(double Angle, double Score)>();

                    foreach(var orientation in scored.Select(sc => sc.Angle).Distinct())
                        rescored.Add((orientation,
                                      OcrUprightResolver.OcrScore(escGray, escDisc, orientation, scratch) ?? 0));

                    if(rescored.Count > 0 && rescored.Max(r => r.Score) > (ranked.Count > 0 ? ranked[0].Score : 0))
                    {
                        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                            Console.Error
                                   .WriteLine($"  {Path.GetFileName(inputPath)}: escalated OCR {ranked.FirstOrDefault().Score:0} -> {rescored.Max(r => r.Score):0}");

                        scored = rescored;
                        ranked = scored.OrderByDescending(sc => sc.Score).ToList();
                    }
                }
            }

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
            Mat? smallColor = null;

            Mat SmallColor()
            {
                if(smallColor is not null) return smallColor;
                smallColor = new Mat();

                if(scale < 1.0)
                    Cv2.Resize(src, smallColor, default, scale, scale, InterpolationFlags.Area);
                else
                    src.CopyTo(smallColor);

                return smallColor;
            }

            // Structure corroboration: an answer chosen by text legibility that also brings
            // the label's straight graphic structure into its printed alignment is confirmed
            // by two independent signals — trustworthy without any model. The converse —
            // strong structure that a marginal winner clearly misaligns — is a classical
            // veto, or better, a classical re-pick.
            //
            // A dominant rectangular badge (a "PC CD-ROM" box, a banner) is the strongest
            // structural cue and works mod 180: badges are printed long-axis HORIZONTAL, so
            // a result that leaves the badge vertical is a 90°-off answer even though it is
            // perfectly "axis-aligned" mod 90. Anonymous segments only support mod 90.
            var rectPick = ranked.Count > 0 && ranked[0].Score > 0
                               ? StructureEstimator.RectanglePick(small, smallDisc)
                               : null;

            if(rectPick is {} pick)
            {
                var rectTheta = pick.Theta;
                var longAxis  = ((rectTheta + angle) % 180 + 180) % 180;
                var offHoriz = Math.Min(longAxis, 180 - longAxis);

                if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                    Console.Error
                           .WriteLine($"  {Path.GetFileName(inputPath)}: rectangle long axis {rectTheta:0.00}° off-horizontal after rotation {offHoriz:0.00}°");

                if(offHoriz <= 1.5 && confidence >= 1.2)
                {
                    confidence = Math.Max(confidence, options.MinConfidence);
                    method     += "+structure";
                }
                else if(offHoriz <= 6 && confidence >= 1.2)
                {
                    // Same flip, small tilt: the text evidence chose the right orientation
                    // but sits a few degrees off (tesseract reads tilted text nearly as well
                    // as straight); the badge measures the exact correction.
                    var a       = ((-rectTheta % 180) + 180) % 180;
                    var aligned = AngularDistance(a, angle) <= AngularDistance(a + 180, angle) ? a : a + 180;

                    angle      = aligned;
                    confidence = Math.Max(confidence, options.MinConfidence);
                    method     += "+structure-align";
                }
                else if(offHoriz > 6 && confidence < 2.5)
                {
                    // The weak winner leaves the badge tilted or vertical. Only two
                    // rotations make it horizontal; when OCR can break that 180° tie the
                    // combined evidence replaces the winner. Otherwise refusing beats
                    // confidently applying a wrong rotation.
                    var a    = ((-rectTheta % 180) + 180) % 180;
                    var pair = new[] { a, a + 180 };

                    // Whole-disc OCR legibility decides the flip, but single samples are
                    // noisy enough to flip a 180° decision and even a summed sample can lie
                    // on logo-art labels. Require two independent scales to agree on the
                    // direction, each summed over three nearby angles; when they disagree,
                    // or the weaker scale is not decisive, refuse instead of guessing.
                    using var smallOcr = new Mat();
                    small.CopyTo(smallOcr);
                    PreprocessForOcr(smallOcr, smallDisc);

                    double PairSum(Mat img, Disc dsc, double p) =>
                        new[] { -1.5, 0.0, 1.5 }
                           .Sum(off => OcrUprightResolver.OcrScore(img, dsc, ((p + off) % 360 + 360) % 360,
                                                                   scratch) ??
                                       0);

                    var ocrA = PairSum(ocrGray, ocrDisc, pair[0]);
                    var ocrB = PairSum(ocrGray, ocrDisc, pair[1]);
                    var detA = PairSum(smallOcr, smallDisc, pair[0]);
                    var detB = PairSum(smallOcr, smallDisc, pair[1]);

                    var sameDirection = ocrA >= ocrB == detA >= detB;
                    var ratioOcr      = Math.Max(ocrA, ocrB) / Math.Max(1, Math.Min(ocrA, ocrB));
                    var ratioDet      = Math.Max(detA, detB) / Math.Max(1, Math.Min(detA, detB));

                    var pairScores = new List<(double Angle, double Score)>
                    {
                        (pair[0], ocrA + detA), (pair[1], ocrB + detB)
                    };

                    pairScores = pairScores.OrderByDescending(pp => pp.Score).ToList();

                    var decisive = sameDirection && Math.Min(ratioOcr, ratioDet) >= 1.35 && pairScores[0].Score > 0;

                    if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                        Console.Error
                               .WriteLine($"  {Path.GetFileName(inputPath)}: rectangle pick {rectTheta:0.00}° -> {pairScores[0].Angle:0.00}° ocr {pairScores[0].Score:0} vs {pairScores[1].Score:0}");

                    if(decisive)
                    {
                        angle      = pairScores[0].Angle;
                        confidence = Math.Max(confidence, options.MinConfidence);
                        method     += "+structure-pick";
                    }
                    else
                    {
                        confidence = Math.Min(confidence, options.MinConfidence * 0.79);
                        method     += "+structure-veto";
                    }
                }
            }
            else if(StructureEstimator.Dominant(small, smallDisc) is {} st && ranked.Count > 0 && ranked[0].Score > 0)
            {
                var aligned = ((st.Orientation + angle) % 90 + 90) % 90;
                var offAxis = Math.Min(aligned, 90 - aligned);

                if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                    Console.Error
                           .WriteLine($"  {Path.GetFileName(inputPath)}: structure peak {st.Orientation:0.00}° sharpness={st.Sharpness:0.00} off-axis after rotation {offAxis:0.00}°");

                if(st.Sharpness >= 0.5 && offAxis <= 1.5 && confidence >= 1.2)
                {
                    confidence = Math.Max(confidence, options.MinConfidence);
                    method     += "+structure";
                }
                else if(st.Sharpness >= 0.65 && offAxis > 6 && confidence < 2.5)
                {
                    confidence = Math.Min(confidence, options.MinConfidence * 0.79);
                    method     += "+structure-veto";
                }
            }

            // Baseline-quorum confidence: on clear-text labels the OCR winner is often the
            // right orientation with a mediocre ratio (other directions read a little) and
            // a tilt of several degrees (tesseract reads tilted text nearly as well as
            // straight). A strong quorum of wide text lines agreeing on one tilt is
            // independent classical evidence: it confirms the axis, measures the exact
            // correction, and leaves OCR only the 180° flip to vouch for.
            if(confidence < options.MinConfidence && ranked.Count > 0 && ranked[0].Score > 0 &&
               !method.EndsWith("+structure-veto", StringComparison.Ordinal) &&
               BaselineDeskew.Residual(ocrGray, ocrDisc, angle, out var quorum) is {} tilt &&
               quorum && Math.Abs(tilt) <= 15)
            {
                var twin = scored.Where(sc => AngularDistance(sc.Angle, angle + 180) < 10)
                                 .Select(sc => sc.Score)
                                 .DefaultIfEmpty(0)
                                 .Max();

                var flipRatio = twin > 0 ? ranked[0].Score / twin : 10.0;

                if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                    Console.Error
                           .WriteLine($"  {Path.GetFileName(inputPath)}: baseline quorum tilt {tilt:0.00}° flip-ratio {flipRatio:0.00}");

                if(flipRatio >= 1.2)
                {
                    angle      = ((angle + tilt) % 360 + 360) % 360;
                    confidence = Math.Max(confidence, options.MinConfidence);
                    method     += "+baseline";
                }
            }

            // Rotation-stability confidence: when the score ratio is indecisive
            // (multi-directional designs read a little text at several orientations), a
            // decisive classical signal remains — re-estimate on the same disc rotated by
            // 37°. If the answer tracks the rotation, the estimator follows real label
            // features, not noise, and the result is trustworthy.
            // The structure veto is final for classical escalation: a projection/OCR
            // estimator locked onto a deliberately tilted text block tracks a probe
            // rotation perfectly — stability must not resurrect what structure refuted.
            if(confidence < options.MinConfidence && ranked.Count > 0 && ranked[0].Score > 0 &&
               !method.EndsWith("+structure-veto", StringComparison.Ordinal) && OcrUprightResolver.IsAvailable)
            {
                const double probe = 37.0;

                using var smallR = Rotated(small, smallDisc.Center, probe);
                using var ocrR   = Rotated(ocrGray, ocrDisc.Center, probe);

                if(BestOrientation(smallR, smallDisc, ocrR, ocrDisc, scratch) is {} second)
                {
                    var tracked = AngularDistance(second, angle - probe) <= 3.0;

                    if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                        Console.Error
                               .WriteLine($"  {Path.GetFileName(inputPath)}: stability probe {second:0.00}° vs expected {((angle - probe) % 360 + 360) % 360:0.00}° -> {(tracked ? "stable" : "unstable")}");

                    if(tracked)
                    {
                        confidence = Math.Max(confidence, options.MinConfidence);
                        method     += "+stable";
                    }
                }
            }

            try
            {
                if(confidence < options.MinConfidence && options.OpenAi.IsUsable)
                {
                    // A dominant rectangle pins the axis classically; the model only ever
                    // gets to break the remaining 180° tie — never to propose an
                    // orientation that leaves the badge tilted or vertical.
                    var orientations = rectPick is {} rp0 && rp0.Theta is {} rp
                                           ? new List<double>
                                           {
                                               ((-rp % 180) + 180) % 180, ((-rp % 180) + 180) % 180 + 180
                                           }
                                           : scored.Count > 0
                                           ? ranked.Take(6).Select(s => s.Angle).ToList()
                                           : candidates.SelectMany(c => new[]
                                                        {
                                                            c.Angle, c.Angle + 180
                                                        })
                                                       .Select(a => ((a % 360) + 360) % 360)
                                                       .ToList();

                    if(OpenAiOrientationResolver.Resolve(SmallColor(), smallDisc, orientations, options.OpenAi) is {}
                       aiPick)
                    {
                        angle      = aiPick;
                        confidence = options.MinConfidence; // the model's choice is applied
                        method     += "+openai";
                    }
                }

                if(confidence >= options.MinConfidence)
                {
                    // Final polish: candidate angles inherit the precision of whichever
                    // estimator proposed them (the implicit 0° and arc candidates only a few
                    // degrees), so re-center on the OCR-legibility maximum ±8°, then fine
                    // projection polish. This removes the residual "almost straight" tilts.
                    //
                    // Drift cap: polish exists for precision, not to re-decide. On designs
                    // with a deliberately slanted text block, legibility keeps improving all
                    // the way to "slanted block horizontal" and the polish chain walks many
                    // degrees away from a decisive coarse winner — a worse answer with better
                    // OCR. If polish moves more than a few degrees, revert to the decided
                    // angle and only fine-tune narrowly around it.
                    var decided  = angle;
                    var polished = RefineWithOcr(ocrGray, ocrDisc, angle, scratch);
                    polished = AngleEstimator.RefineAround(ocrGray, ocrDisc, polished, window: 6.0, step: 0.5);

                    if(AngularDistance(polished, decided) <= 4.0)
                        angle = polished;
                    else
                        angle = AngleEstimator.RefineAround(ocrGray, ocrDisc, decided, window: 2.0, step: 0.25);

                    // Sub-degree finish: measure the residual tilt from the text baselines
                    // themselves at OCR resolution; the projection polish alone bottoms out
                    // around ±1° on sparse-text labels.
                    if(BaselineDeskew.Residual(ocrGray, ocrDisc, angle, out var strongQuorum) is {} residual)
                    {
                        if(Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null)
                            Console.Error
                                   .WriteLine($"  {Path.GetFileName(inputPath)}: pre-deskew {angle:0.00}° residual {residual:0.00}°{(strongQuorum ? " (strong quorum)" : "")}");

                        // A strong quorum of agreeing text lines may correct well beyond the
                        // sub-degree range: an OCR-chosen winner sits many degrees off on
                        // clear-text labels because tesseract reads tilted text nearly as
                        // well as straight text.
                        var limit = strongQuorum ? 15.0 : 3.0;

                        if(Math.Abs(residual) <= limit) angle = ((angle + residual) % 360 + 360) % 360;
                    }

                    // Vision verification: for anything short of overwhelming confidence, show
                    // the corrected label to the model; on NO, walk the next distinct-ranked
                    // orientations until one verifies. Prevents the "completely wrong quadrant"
                    // failures a wrong OCR/arc winner produces.
                    if(confidence < options.VerifyBelow && options.OpenAi.IsUsable)
                    {
                        var verdict = OpenAiOrientationResolver.VerifyUpright(SmallColor(), smallDisc, angle,
                                                                              options.OpenAi);

                        if(verdict is false)
                        {
                            var accepted = false;

                            // With a dominant rectangle, only alternatives that keep the
                            // badge horizontal are acceptable — the model must not walk the
                            // answer onto a structure-misaligned orientation.
                            foreach(var alt in ranked.Where(r => r.Score > 0 &&
                                                                 AngularDistance(r.Angle, angle) > 20 &&
                                                                 (rectPick is not {} rp2 ||
                                                                  Math.Min(((rp2.Theta + r.Angle) % 180 + 180) % 180,
                                                                           180 -
                                                                           ((rp2.Theta + r.Angle) % 180 + 180) %
                                                                           180) <=
                                                                  6))
                                                     .Select(r => r.Angle)
                                                     .Take(3))
                            {
                                var a2 = RefineWithOcr(ocrGray, ocrDisc, alt, scratch);
                                a2 = AngleEstimator.RefineAround(small, smallDisc, a2);

                                if(OpenAiOrientationResolver.VerifyUpright(SmallColor(), smallDisc, a2,
                                                                           options.OpenAi) is not true)
                                    continue;

                                angle    = a2;
                                method   += "+veto";
                                accepted = true;

                                break;
                            }

                            // Nothing verified: better to leave the scan untouched than to
                            // apply a rotation the model rejects.
                            if(!accepted) confidence = 0;
                        }
                    }
                }
            }
            finally
            {
                smallColor?.Dispose();
            }
        }

        return Finish(inputPath, outputPath, options, src, fullDisc, small, smallDisc, angle, confidence, method);
    }

    /// <summary>Common tail of <see cref="ProcessFile"/>: debug artifacts, rotation, write, metadata, result.</summary>
    private static AngleResult Finish(string inputPath, string outputPath, Options options, Mat src, Disc fullDisc,
                                      Mat small, Disc smallDisc, double angle, double confidence, string method)
    {
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