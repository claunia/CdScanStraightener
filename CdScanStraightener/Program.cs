using System.CommandLine;
using CdScanStraightener.Cli;
using CdScanStraightener.Io;

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
    };

    return BatchRunner.RunAsync(options, ct);
});

return await root.Parse(args).InvokeAsync();