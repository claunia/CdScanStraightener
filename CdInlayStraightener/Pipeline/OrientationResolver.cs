using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

public sealed record Orientation(int Quarter, double Confidence);

/// <summary>
/// Coarse orientation for a deskewed inlay: OCR the crop at 0/90/180/270 and keep the most
/// legible. Applying a 90° multiple uses Cv2.Rotate — lossless, no resampling.
/// </summary>
public static class OrientationResolver
{
    private const int OcrMaxSide = 1536;

    public static Orientation Resolve(Mat crop, string scratchDir)
    {
        if(!TesseractOcr.IsAvailable) return new Orientation(0, 0);

        using var gray = new Mat();
        Cv2.CvtColor(crop, gray, ColorConversionCodes.BGR2GRAY);

        var       scale = Math.Min(1.0, (double)OcrMaxSide / Math.Max(crop.Width, crop.Height));
        using var small = new Mat();

        if(scale < 1.0)
            Cv2.Resize(gray, small, default, scale, scale, InterpolationFlags.Area);
        else
            gray.CopyTo(small);

        var scores = new double[4];

        for(var q = 0; q < 4; q++)
        {
            using var view = Rotate(small, q);
            var       tmp  = Path.Combine(scratchDir, $"inlay-{Guid.NewGuid():N}.png");

            try
            {
                Cv2.ImWrite(tmp, view);
                scores[q] = TesseractOcr.RunTsv(tmp) is {} words ? TesseractOcr.Score(words) : 0;
            }
            finally
            {
                if(File.Exists(tmp)) File.Delete(tmp);
            }
        }

        var best = Array.IndexOf(scores, scores.Max());

        if(scores[best] <= 0) return new Orientation(0, 0);

        var second = scores.Where((_, i) => i != best).Max();

        return new Orientation(best * 90, second > 0 ? scores[best] / second : 10.0);
    }

    /// <summary>Lossless rotation by quarter turns (0–3 counterclockwise).</summary>
    public static Mat Rotate(Mat src, int quarters)
    {
        var dst = new Mat();

        switch(((quarters % 4) + 4) % 4)
        {
            case 0:
                src.CopyTo(dst);

                break;
            case 1:
                Cv2.Rotate(src, dst, RotateFlags.Rotate90Counterclockwise);

                break;
            case 2:
                Cv2.Rotate(src, dst, RotateFlags.Rotate180);

                break;
            default:
                Cv2.Rotate(src, dst, RotateFlags.Rotate90Clockwise);

                break;
        }

        return dst;
    }
}
