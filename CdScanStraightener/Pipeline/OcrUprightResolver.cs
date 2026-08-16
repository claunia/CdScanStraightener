using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Disc-specific OCR scoring: renders the disc at a candidate rotation and measures how
/// legibly tesseract reads it. Tesseract plumbing lives in <see cref="TesseractOcr"/>.
/// </summary>
public static class OcrUprightResolver
{
    public static bool IsAvailable => TesseractOcr.IsAvailable;

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
            var words = TesseractOcr.RunTsv(tmp);

            if(words is null) return null;

            return TesseractOcr.Score(words);
        }
        finally
        {
            if(File.Exists(tmp)) File.Delete(tmp);
        }
    }
}
