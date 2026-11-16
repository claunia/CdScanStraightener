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

        var dpi = PngMetadata.ReadDpi(inputPath);

        if(options.Box) return ProcessBox(src, dpi, inputPath, outputPath, options);

        // A known console cover layout fixes geometry and orientation outright.
        using(var wii = WiiCover.Detect(src, dpi))
        {
            if(wii is not null)
            {
                if(!options.DryRun)
                {
                    Cv2.ImWrite(outputPath, wii.Image);
                    PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
                }

                return new InlayResult(Path.GetFileName(inputPath), wii.Skew, wii.Quarters * 90,
                                       double.PositiveInfinity, wii.Image.Width, wii.Image.Height, true);
            }
        }

        var inlay = InlayDetector.Detect(src, dpi);

        if(inlay is null) return Untouched(src, inputPath, outputPath, options);

        var skew = inlay.Skew;

        using var cropped = Cropper.DeskewAndCrop(src, inlay, options.Trim);

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
    /// <summary>Nothing inlay-sized found: copy untouched, flag in the report.</summary>
    private static InlayResult Untouched(Mat src, string inputPath, string outputPath, Options options)
    {
        if(!options.DryRun)
        {
            Cv2.ImWrite(outputPath, src);
            PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
        }

        return new InlayResult(Path.GetFileName(inputPath), 0, 0, 0, src.Width, src.Height, false);
    }

    /// <summary>
    /// A retail box face: outline as for inlays, then inward past the side-wall smear to the
    /// face itself; turned to the requested aspect, then upright or upside down by OCR, or by
    /// a known logo when the face has too little text.
    /// </summary>
    private static InlayResult ProcessBox(Mat src, double? dpi, string inputPath, string outputPath, Options options)
    {
        var inlay = InlayDetector.Detect(src, dpi);

        if(inlay is null) return Untouched(src, inputPath, outputPath, options);

        using var deskewed = Cropper.Deskew(src, inlay);

        var face = BoxFaceRefiner.Refine(deskewed, inlay.Box, dpi, BoxFaceRefiner.BackgroundLuminance(src));

        if(options.Verbose)
            Console.Error.WriteLine($"    {Path.GetFileName(inputPath)}: face inset L {face.InsetMm[0]:0.0} " +
                                    $"T {face.InsetMm[1]:0.0} R {face.InsetMm[2]:0.0} B {face.InsetMm[3]:0.0} mm");

        using var cropped = Cropper.Crop(deskewed, face.Box, options.Trim);

        var orientation = options.ForceOrientation is {} forced
                              ? new Orientation(forced, double.PositiveInfinity)
                              : BoxOrientation(cropped, dpi, options);

        using var final = OrientationResolver.Rotate(cropped, orientation.Quarter / 90);

        if(!options.DryRun)
        {
            Cv2.ImWrite(outputPath, final);
            PngMetadata.WritePreservedChunks(outputPath, PngMetadata.ReadPreservedChunks(inputPath));
        }

        return new InlayResult(Path.GetFileName(inputPath), inlay.Skew, orientation.Quarter, orientation.Confidence,
                               final.Width, final.Height, true);
    }

    /// <summary>Logos are matched at about the scale the disc straightener sees them.</summary>
    private const double LogoPxPerMm = 1536 / 120.0;

    /// <summary>
    /// The face's aspect is known (portrait unless --landscape), which leaves two quarter turns.
    /// A known logo printed upright settles it first — box fronts can carry almost no text, and
    /// OCR then reads artwork confidently the wrong way up. Otherwise OCR picks: the choice is
    /// binary and has to be made, so even a weak preference beats a guess. With neither the face
    /// is only turned to the right aspect.
    /// </summary>
    private static Orientation BoxOrientation(Mat cropped, double? dpi, Options options)
    {
        var portrait   = cropped.Height >= cropped.Width;
        int[] quarters = portrait != options.Landscape ? [0, 2] : [1, 3];
        var debug      = Environment.GetEnvironmentVariable("CDSCAN_DEBUG") is not null;

        using(var view = OrientationResolver.Rotate(cropped, quarters[0]))
        using(var gray = new Mat())
        using(var small = new Mat())
        {
            Cv2.CvtColor(view, gray, ColorConversionCodes.BGR2GRAY);

            var scale = Math.Min(1.0, LogoPxPerMm * 25.4 / (dpi ?? 600));
            Cv2.Resize(gray, small, default, scale, scale, InterpolationFlags.Area);

            // Box art prints logos in either polarity (a white apple on black, a grey one on
            // white), and keypoint descriptors do not survive inversion: try both.
            using var inverted = new Mat();
            Cv2.BitwiseNot(small, inverted);

            var logo = new[] { LogoAnchor.Resolve(small), LogoAnchor.Resolve(inverted) }.Where(m => m is not null)
                                                                                         .MaxBy(m => m!.Inliers);

            if(logo is not null)
            {
                if(debug) Console.Error.WriteLine($"    box logo: {logo}");

                var off = Math.Abs(((logo.Angle + 180) % 360 + 360) % 360 - 180); // 0 upright, 180 upside down

                if(off <= 20) return new Orientation(quarters[0] * 90, double.PositiveInfinity);

                if(off >= 160) return new Orientation(quarters[1] * 90, double.PositiveInfinity);
            }

            var pxPerMm = scale * (dpi ?? 600) / 25.4;

            if(SilhouetteAnchor.Resolve(small, (int)(5 * pxPerMm), (int)(60 * pxPerMm)) is {} silhouette)
            {
                if(debug) Console.Error.WriteLine($"    box silhouette logo: {silhouette}°");

                return new Orientation(quarters[silhouette == 0 ? 0 : 1] * 90, double.PositiveInfinity);
            }
        }

        var ocr = OrientationResolver.Resolve(cropped, Path.GetTempPath(), quarters);

        if(debug) Console.Error.WriteLine($"    box OCR among [{string.Join(',', quarters)}]: {ocr}");

        return ocr.Confidence > 1 && quarters.Contains(ocr.Quarter / 90)
                   ? ocr
                   : new Orientation(quarters[0] * 90, 0);
    }
}
