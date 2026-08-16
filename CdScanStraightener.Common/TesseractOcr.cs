using System.Diagnostics;
using System.Globalization;

namespace CdScanStraightener.Common;

public sealed record OcrWord(double Conf, string Text, int X, int Y, int W, int H);

/// <summary>Shared tesseract invocation: availability, language handling and TSV parsing.</summary>
public static class TesseractOcr
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

    /// <summary>Tesseract languages ("eng", "eng+spa", …). Null = all installed packs.</summary>
    public static string? LanguageOverride { get; set; }

    // All installed language packs (minus special ones): label text is often not English,
    // and OCRing with the right language dramatically improves legibility discrimination.
    private static readonly Lazy<string> Languages = new(() =>
    {
        try
        {
            var psi = new ProcessStartInfo("tesseract", "--list-langs")
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true
            };

            using var proc = Process.Start(psi);

            if(proc is null) return "eng";
            var output = proc.StandardOutput.ReadToEnd() + proc.StandardError.ReadToEnd();
            proc.WaitForExit(5000);

            var langs = output.Split('\n')
                              .Select(l => l.Trim())
                              .Where(l => l.Length == 3 && l.All(char.IsAsciiLetterLower))
                              .Where(l => l != "osd" && l != "equ" && l != "snum")
                              .ToArray();

            return langs.Length > 0 ? string.Join('+', langs) : "eng";
        }
        catch
        {
            return "eng";
        }
    });

    /// <summary>
    /// Runs tesseract on an image file and returns the confidently-read words with their
    /// bounding boxes, or null if tesseract failed. Only words with conf ≥ 40, at least two
    /// characters and some letter/digit content are returned.
    /// </summary>
    public static List<OcrWord>? RunTsv(string imagePath)
    {
        var psi = new ProcessStartInfo("tesseract",
                                       $"\"{imagePath}\" stdout --psm 11 -l {LanguageOverride ?? Languages.Value} tsv")
        {
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
        };

        // Callers already parallelize per image; tesseract's own OpenMP threads only oversubscribe.
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

        // TSV columns: ... left(6) top(7) width(8) height(9) conf(10) text(11).
        var words = new List<OcrWord>();

        foreach(var line in output.Split('\n').Skip(1))
        {
            var cols = line.Split('\t');

            if(cols.Length < 12) continue;
            if(!double.TryParse(cols[10], NumberStyles.Float, CultureInfo.InvariantCulture, out var conf)) continue;
            var text = cols[11].Trim();

            if(conf < 40 || text.Length < 2 || !text.Any(char.IsLetterOrDigit)) continue;

            if(int.TryParse(cols[6], out var x) &&
               int.TryParse(cols[7], out var y) &&
               int.TryParse(cols[8], out var w) &&
               int.TryParse(cols[9], out var h))
                words.Add(new OcrWord(conf, text, x, y, w, h));
        }

        return words;
    }

    /// <summary>Legibility score used across the tools: Σ confidence × letter count.</summary>
    public static double Score(IEnumerable<OcrWord> words) =>
        words.Sum(w => w.Conf * w.Text.Count(char.IsLetterOrDigit));
}
