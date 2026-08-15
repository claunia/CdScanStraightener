using System.CommandLine;
using CdScanStraightener.Cli;
using CdScanStraightener.Io;
using Microsoft.Extensions.Configuration;

// Optional vision-model fallback settings: "OpenAI" section of appsettings.json next to
// the executable or in the working directory; environment variables (OpenAI__ApiKey, …)
// override. Works with any OpenAI-compatible endpoint, e.g. LM Studio.
var configuration = new ConfigurationBuilder().SetBasePath(AppContext.BaseDirectory)
                                              .AddJsonFile("appsettings.json", optional: true)
                                              .AddJsonFile(Path.Combine(Environment.CurrentDirectory,
                                                                        "appsettings.json"),
                                                           optional: true)
                                              .AddEnvironmentVariables()
                                              .Build();

var openAiSettings = configuration.GetSection("OpenAI").Get<OpenAiSettings>() ?? new OpenAiSettings();

var inputOpt = new Option<DirectoryInfo>("--input", "-i")
{
    Description = "Folder containing the scanned CD PNGs.",
    Required    = true,
};

var outputOpt = new Option<DirectoryInfo>("--output", "-o")
{
    Description = "Folder to write straightened PNGs to.",
    Required    = true,
};

var dryRunOpt = new Option<bool>("--dry-run")
{
    Description = "Detect and report angles without writing any output files.",
};

var verboseOpt = new Option<bool>("--verbose", "-v")
{
    Description = "Print detection details per file.",
};

var reportOpt = new Option<FileInfo?>("--report")
{
    Description = "Write a CSV report of detected angles and confidences.",
};

var forceAngleOpt = new Option<double?>("--force-angle")
{
    Description = "Skip detection and rotate every image by this angle (degrees, counterclockwise).",
};

var debugDirOpt = new Option<DirectoryInfo?>("--debug-dir")
{
    Description = "Write annotated detection intermediates (disc overlay, polar unwrap) here.",
};

var minConfidenceOpt = new Option<double>("--min-confidence")
{
    Description         = "Below this detection confidence the image is copied unrotated with a warning.",
    DefaultValueFactory = _ => 1.5,
};

var ocrLangsOpt = new Option<string>("--ocr-langs")
{
    Description         = "Tesseract language(s) for orientation OCR, e.g. \"eng\" or \"eng+spa\" (default: all installed).",
    DefaultValueFactory = _ => "auto",
};

var centerOpt = new Option<bool>("--center")
{
    Description = "Center the disc on a white square canvas (disc diameter + safe area per side); " +
                  "everything outside the disc becomes white. Changes output dimensions.",
};

var safeAreaOpt = new Option<int>("--safe-area")
{
    Description         = "White margin in pixels around the disc when using --center.",
    DefaultValueFactory = _ => 25,
};

var verifyBelowOpt = new Option<double>("--verify-below")
{
    Description         = "Vision-verify results whose confidence is below this value; use a large number " +
                          "to verify every image (slow but thorough).",
    DefaultValueFactory = _ => 3.0,
};

var overwriteOpt = new Option<bool>("--overwrite")
{
    Description = "Overwrite existing files in the output folder (default: skip them).",
};

var root =
    new RootCommand("Straightens scanned CD images: detects the disc and the rotation that makes the label text " +
                    "read upright, then rotates in place (same canvas size, DPI and ICC profile preserved).")
    {
        inputOpt,
        outputOpt,
        dryRunOpt,
        verboseOpt,
        reportOpt,
        forceAngleOpt,
        debugDirOpt,
        minConfidenceOpt,
        overwriteOpt,
        ocrLangsOpt,
        centerOpt,
        safeAreaOpt,
        verifyBelowOpt,
    };

root.SetAction((parseResult, ct) =>
{
    var options = new Options
    {
        Input         = parseResult.GetValue(inputOpt)!,
        Output        = parseResult.GetValue(outputOpt)!,
        DryRun        = parseResult.GetValue(dryRunOpt),
        Verbose       = parseResult.GetValue(verboseOpt),
        Report        = parseResult.GetValue(reportOpt),
        ForceAngle    = parseResult.GetValue(forceAngleOpt),
        DebugDir      = parseResult.GetValue(debugDirOpt),
        MinConfidence = parseResult.GetValue(minConfidenceOpt),
        Overwrite     = parseResult.GetValue(overwriteOpt),
        OcrLangs      = parseResult.GetValue(ocrLangsOpt)!,
        Center        = parseResult.GetValue(centerOpt),
        SafeArea      = parseResult.GetValue(safeAreaOpt),
        VerifyBelow   = parseResult.GetValue(verifyBelowOpt),
        OpenAi        = openAiSettings,
    };

    return BatchRunner.RunAsync(options, ct);
});

return await root.Parse(args).InvokeAsync();