using CdInlayStraightener.Cli;
using CdScanStraightener.Common;
using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

public sealed record InlayResult(
    string File,
    double Skew,
    int    Orientation,
    double Confidence,
    int    Width,
    int    Height,
    bool   Detected);

public static class InlayPipeline
{
    public static InlayResult ProcessFile(string inputPath, string outputPath, Options options)
    {
        using var src = Cv2.ImRead(inputPath, ImreadModes.Color);

        if(src.Empty()) throw new InvalidDataException($"Could not read image: {inputPath}");

        var inlay = InlayDetector.Detect(src);

        if(inlay is null)
        {
            // Nothing inlay-sized found: copy untouched, flag in the report.
            if(!options.DryRun)
            {
                Cv2.ImWrite(outputPath, src);
                PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
            }

            return new InlayResult(Path.GetFileName(inputPath), 0, 0, 0, src.Width, src.Height, false);
        }

        var skew = InlayDetector.Skew(inlay.Rect);

        using var cropped = Cropper.DeskewAndCrop(src, inlay, skew, options.Trim);

        var orientation = options.ForceOrientation is {} forced
                              ? new Orientation(forced, double.PositiveInfinity)
                              : OrientationResolver.Resolve(cropped, Path.GetTempPath());

        var apply = orientation.Confidence >= options.MinConfidence;

        using var final = apply ? OrientationResolver.Rotate(cropped, orientation.Quarter / 90) : cropped.Clone();

        if(!options.DryRun)
        {
            Cv2.ImWrite(outputPath, final);
            PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
        }

        return new InlayResult(Path.GetFileName(inputPath),
                               skew,
                               apply ? orientation.Quarter : 0,
                               orientation.Confidence,
                               final.Width,
                               final.Height,
                               true);
    }
}
