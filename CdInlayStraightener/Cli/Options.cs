namespace CdInlayStraightener.Cli;

public sealed record Options
{
    public required DirectoryInfo Input         { get; init; }
    public required DirectoryInfo Output        { get; init; }
    public          bool          DryRun        { get; init; }
    public          bool          Verbose       { get; init; }
    public          FileInfo?     Report        { get; init; }
    public          int           Trim          { get; init; } = 2;
    public          double        MinConfidence { get; init; } = 1.5;
    public          int?          ForceOrientation { get; init; }
    public          string        OcrLangs      { get; init; } = "auto";
    public          bool          Overwrite     { get; init; }
}
