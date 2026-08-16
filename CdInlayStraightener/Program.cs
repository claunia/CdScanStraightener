using System.CommandLine;
using CdInlayStraightener.Cli;
using CdInlayStraightener.Io;

var inputOpt = new Option<DirectoryInfo>("--input", "-i")
{
    Description = "Folder containing the scanned inlay PNGs.",
    Required    = true,
};

var outputOpt = new Option<DirectoryInfo>("--output", "-o")
{
    Description = "Folder to write straightened, cropped PNGs to.",
    Required    = true,
};

var dryRunOpt = new Option<bool>("--dry-run")
{
    Description = "Detect and report without writing any output files.",
};

var verboseOpt = new Option<bool>("--verbose", "-v")
{
    Description = "Print detection details per file.",
};

var reportOpt = new Option<FileInfo?>("--report")
{
    Description = "Write a CSV report of skew, orientation and crop sizes.",
};

var trimOpt = new Option<int>("--trim")
{
    Description         = "Pixels trimmed inward from the detected edges so no background survives.",
    DefaultValueFactory = _ => 2,
};

var minConfidenceOpt = new Option<double>("--min-confidence")
{
    Description         = "Below this OCR-orientation confidence the inlay is kept as deskewed (no 90°/180° turn).",
    DefaultValueFactory = _ => 1.5,
};

var forceOrientationOpt = new Option<int?>("--force-orientation")
{
    Description = "Skip OCR and turn every inlay by this many degrees counterclockwise (0, 90, 180 or 270).",
};

var ocrLangsOpt = new Option<string>("--ocr-langs")
{
    Description         = "Tesseract language(s) for orientation OCR, e.g. \"eng\" or \"eng+spa\" (default: all installed).",
    DefaultValueFactory = _ => "auto",
};

var overwriteOpt = new Option<bool>("--overwrite")
{
    Description = "Overwrite existing files in the output folder (default: skip them).",
};

var root = new RootCommand("Straightens scanned CD inlays/booklets: detects the rectangle on a light or dark " +
                           "scanner background, deskews it, fixes 90°/180° orientation via OCR (losslessly), and " +
                           "crops so no background pixels remain. DPI and ICC metadata are preserved.")
{
    inputOpt,
    outputOpt,
    dryRunOpt,
    verboseOpt,
    reportOpt,
    trimOpt,
    minConfidenceOpt,
    forceOrientationOpt,
    ocrLangsOpt,
    overwriteOpt,
};

root.SetAction((parseResult, ct) =>
{
    var forced = parseResult.GetValue(forceOrientationOpt);

    if(forced is not (null or 0 or 90 or 180 or 270))
    {
        Console.Error.WriteLine("--force-orientation must be 0, 90, 180 or 270.");

        return Task.FromResult(1);
    }

    var options = new Options
    {
        Input            = parseResult.GetValue(inputOpt)!,
        Output           = parseResult.GetValue(outputOpt)!,
        DryRun           = parseResult.GetValue(dryRunOpt),
        Verbose          = parseResult.GetValue(verboseOpt),
        Report           = parseResult.GetValue(reportOpt),
        Trim             = parseResult.GetValue(trimOpt),
        MinConfidence    = parseResult.GetValue(minConfidenceOpt),
        ForceOrientation = forced,
        OcrLangs         = parseResult.GetValue(ocrLangsOpt)!,
        Overwrite        = parseResult.GetValue(overwriteOpt),
    };

    return BatchRunner.RunAsync(options, ct);
});

return await root.Parse(args).InvokeAsync();
