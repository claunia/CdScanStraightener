using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Orientation anchoring by known-logo matching: rating squares (USK/PEGI), platform
/// wordmarks (Wii, Wii U, Nintendo), media badges (COMPACT disc, DVD, PC CD-ROM) and
/// similar marks are printed upright on the label, and one confident match pins the
/// full 360° orientation with sub-degree precision — no OCR, no 180° ambiguity.
///
/// Matching is ORB keypoints + RANSAC similarity transform: keypoints are detected on an
/// image pyramid and carry their own orientation, so the match survives different
/// scanners, resolutions and rotations; the transform's scale component measures the DPI
/// difference instead of assuming it. Templates are tried at several pre-scales because
/// ORB's own pyramid only covers a limited scale range.
/// </summary>
public static class LogoAnchor
{
    public sealed record Match(string Template, double Angle, int Inliers, double Scale);

    private sealed record Template(string Name, Mat Image, KeyPoint[] Keypoints, Mat Descriptors);

    private static readonly Lazy<List<Template>> Templates = new(LoadTemplates);

    public static bool IsAvailable => Templates.Value.Count > 0;

    private static List<Template> LoadTemplates()
    {
        var list = new List<Template>();

        var dirs = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Templates"), Path.Combine(Environment.CurrentDirectory, "Templates")
        };

        var dir = dirs.FirstOrDefault(Directory.Exists);

        if(dir is null) return list;

        using var orb = ORB.Create(nFeatures: 1500);

        foreach(var file in Directory.EnumerateFiles(dir, "*.png").OrderBy(f => f))
        {
            var img = Cv2.ImRead(file, ImreadModes.Grayscale);

            if(img.Empty() || Math.Min(img.Width, img.Height) < 24)
            {
                img.Dispose();

                continue;
            }

            var desc = new Mat();
            orb.DetectAndCompute(img, null, out var kp, desc);

            if(kp.Length < 30)
            {
                img.Dispose();
                desc.Dispose();

                continue;
            }

            list.Add(new Template(Path.GetFileNameWithoutExtension(file), img, kp, desc));
        }

        return list;
    }

    /// <summary>
    /// Matches all templates against the (unpreprocessed) grayscale label and returns the
    /// counterclockwise correction that makes the best-matched logo upright, or null when
    /// nothing matches confidently. <paramref name="minInliers"/> is the RANSAC inlier
    /// count a match must reach — true matches on these labels score 40+, false ones
    /// rarely exceed single digits.
    /// </summary>
    public static Match? Resolve(Mat gray, int minInliers = 12)
    {
        if(!IsAvailable) return null;

        using var orb       = ORB.Create(nFeatures: 12000);
        using var sceneDesc = new Mat();
        orb.DetectAndCompute(gray, null, out var sceneKp, sceneDesc);

        if(sceneKp.Length < 50) return null;

        using var matcher = new BFMatcher(NormTypes.Hamming);
        Match?    best    = null;

        foreach(var template in Templates.Value)
        {
            // Templates were cropped from ~1500 px scans; scenes are normalized to a
            // similar size, but high-DPI sources shrink the logos when downscaled, so try
            // smaller template renditions too.
            foreach(var preScale in new[] { 1.0, 0.66, 0.5, 0.33 })
            {
                KeyPoint[] tKp;
                Mat        tDesc;
                Mat?       scaled = null;

                if(Math.Abs(preScale - 1.0) < 1e-9)
                {
                    tKp   = template.Keypoints;
                    tDesc = template.Descriptors;
                }
                else
                {
                    scaled = new Mat();
                    Cv2.Resize(template.Image, scaled, default, preScale, preScale, InterpolationFlags.Area);

                    if(Math.Min(scaled.Width, scaled.Height) < 24)
                    {
                        scaled.Dispose();

                        continue;
                    }

                    using var torb = ORB.Create(nFeatures: 1500);
                    tDesc = new Mat();
                    torb.DetectAndCompute(scaled, null, out tKp, tDesc);
                }

                try
                {
                    if(tKp.Length < 20) continue;

                    var knn = matcher.KnnMatch(tDesc, sceneDesc, 2);

                    var good = knn.Where(m => m.Length == 2 && m[0].Distance < 0.75 * m[1].Distance)
                                  .Select(m => m[0])
                                  .ToArray();

                    if(good.Length < 8) continue;

                    var src = good.Select(m => new Point2f(tKp[m.QueryIdx].Pt.X, tKp[m.QueryIdx].Pt.Y)).ToArray();

                    var dst = good.Select(m => new Point2f(sceneKp[m.TrainIdx].Pt.X, sceneKp[m.TrainIdx].Pt.Y))
                                  .ToArray();

                    using var inlierMask = new Mat();

                    using var transform = Cv2.EstimateAffinePartial2D(InputArray.Create(src),
                                                                      InputArray.Create(dst),
                                                                      inlierMask,
                                                                      RobustEstimationAlgorithms.RANSAC,
                                                                      4.0,
                                                                      5000,
                                                                      0.99,
                                                                      20);

                    if(transform.Empty()) continue;

                    inlierMask.GetArray(out byte[] flags);
                    var inliers = flags.Count(f => f != 0);

                    if(inliers < minInliers) continue;

                    var a     = transform.At<double>(0, 0);
                    var b     = transform.At<double>(1, 0);
                    var scale = Math.Sqrt(a * a + b * b) / preScale;

                    // Similarity sanity: a real print of the same logo lands within a
                    // credible physical size range of the template.
                    if(scale is < 0.2 or > 3.0) continue;

                    // The rotation of the template into the scene, in this codebase's
                    // rotation convention, is directly the correction to apply (verified
                    // against known-angle discs).
                    var angle = Math.Atan2(b, a) * 180 / Math.PI;

                    if(best is null || inliers > best.Inliers)
                        best = new Match(template.Name, ((angle % 360) + 360) % 360, inliers, scale);
                }
                finally
                {
                    if(scaled is not null)
                    {
                        tDesc.Dispose();
                        scaled.Dispose();
                    }
                }
            }
        }

        return best;
    }
}
