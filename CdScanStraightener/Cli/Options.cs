namespace CdScanStraightener.Cli;

public sealed record Options
{
    public required DirectoryInfo  Input         { get; init; }
    public required DirectoryInfo  Output        { get; init; }
    public          bool           DryRun        { get; init; }
    public          bool           Verbose       { get; init; }
    public          FileInfo?      Report        { get; init; }
    public          double?        ForceAngle    { get; init; }
    public          DirectoryInfo? DebugDir      { get; init; }
    public          double         MinConfidence { get; init; } = 1.5;
    public          bool           Overwrite     { get; init; }
    public          OpenAiSettings OpenAi        { get; init; } = new();
    public          string         OcrLangs      { get; init; } = "auto";
}