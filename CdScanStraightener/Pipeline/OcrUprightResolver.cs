using System.Diagnostics;
using System.Globalization;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// 180° disambiguation via the system `tesseract` binary: OCR the label at the candidate
/// angle and at +180°, and keep the orientation with the higher total word confidence.
/// </summary>
public static class OcrUprightResolver
{
    private static readonly Lazy<bool> Available = new(() =>
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo("tesseract", "--version")
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true
            });

            p!.WaitForExit(5000);

            return p.ExitCode == 0;
        }
        catch
        {
            return false;
        }
    });

    public static bool IsAvailable => Available.Value;

    /// <summary>Returns angle or angle+180 (normalized to [0,360)), or null if OCR was inconclusive.</summary>
    public static double? Resolve(Mat gray, Disc disc, double angle, string scratchDir)
    {
        var scoreA = OcrScore(gray, disc, angle,       scratchDir);
        var scoreB = OcrScore(gray, disc, angle + 180, scratchDir);

        if(scoreA is null || scoreB is null || Math.Abs(scoreA.Value - scoreB.Value) < 1e-6) return null;
        var chosen = scoreA > scoreB ? angle : angle + 180;

        return ((chosen % 360) + 360) % 360;
    }

    /// <summary>Legibility score of the label OCRed after rotating by <paramref name="angle"/>; null on OCR failure.</summary>
    public static double? OcrScore(Mat gray, Disc disc, double angle, string scratchDir)
    {
        using var rot     = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(gray, rotated, rot, gray.Size(), InterpolationFlags.Linear, BorderTypes.Constant, Scalar.Black);

        var x0 = Math.Max(0, (int)(disc.Center.X            - disc.Radius));
        var y0 = Math.Max(0, (int)(disc.Center.Y            - disc.Radius));
        var x1 = Math.Min(rotated.Cols, (int)(disc.Center.X + disc.Radius));
        var y1 = Math.Min(rotated.Rows, (int)(disc.Center.Y + disc.Radius));

        if(x1 - x0 < 16 || y1 - y0 < 16) return null;
        using var crop = rotated[new Rect(x0, y0, x1 - x0, y1 - y0)];

        var tmp = Path.Combine(scratchDir, $"ocr-{Guid.NewGuid():N}.png");

        try
        {
            Cv2.ImWrite(tmp, crop);

            var psi = new ProcessStartInfo("tesseract", $"\"{tmp}\" stdout --psm 11 -l eng tsv")
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
            };

            // We already parallelize per image; tesseract's own OpenMP threads only oversubscribe.
            psi.Environment["OMP_THREAD_LIMIT"] = "1";
            using var p = Process.Start(psi);

            if(p is null) return null;
            var output = p.StandardOutput.ReadToEnd();

            if(!p.WaitForExit(30000))
            {
                p.Kill(true);

                return null;
            }

            if(p.ExitCode != 0) return null;

            // TSV columns: ... conf(10) text(11). Sum confidence of confidently-read words.
            double score = 0;

            foreach(var line in output.Split('\n').Skip(1))
            {
                var cols = line.Split('\t');

                if(cols.Length < 12) continue;
                if(!double.TryParse(cols[10], NumberStyles.Float, CultureInfo.InvariantCulture, out var conf)) continue;
                var text = cols[11].Trim();

                if(conf >= 40 && text.Length >= 2 && text.Any(char.IsLetterOrDigit))
                    score += conf * text.Count(char.IsLetterOrDigit);
            }

            return score;
        }
        finally
        {
            if(File.Exists(tmp)) File.Delete(tmp);
        }
    }
}