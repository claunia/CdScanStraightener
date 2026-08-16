using System.Collections.Concurrent;
using System.Globalization;
using CdInlayStraightener.Cli;
using CdInlayStraightener.Pipeline;
using CdScanStraightener.Common;

namespace CdInlayStraightener.Io;

public static class BatchRunner
{
    public static async Task<int> RunAsync(Options options, CancellationToken ct)
    {
        if(!options.Input.Exists)
        {
            Console.Error.WriteLine($"Input directory not found: {options.Input.FullName}");

            return 1;
        }

        var files = options.Input.EnumerateFiles("*.png", SearchOption.TopDirectoryOnly)
                           .OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase)
                           .ToArray();

        if(files.Length == 0)
        {
            Console.Error.WriteLine($"No PNG files in {options.Input.FullName}");

            return 1;
        }

        if(!options.DryRun) options.Output.Create();

        if(!string.Equals(options.OcrLangs, "auto", StringComparison.OrdinalIgnoreCase))
            TesseractOcr.LanguageOverride = options.OcrLangs;

        var       results  = new ConcurrentBag<InlayResult>();
        var       failures = new ConcurrentBag<(string File, string Error)>();
        var       done     = 0;
        using var progress = new ProgressBar(files.Length);

        await Parallel.ForEachAsync(files,
                                    new ParallelOptions
                                    {
                                        MaxDegreeOfParallelism = Environment.ProcessorCount,
                                        CancellationToken      = ct
                                    },
                                    (file, _) =>
                                    {
                                        var outputPath = Path.Combine(options.Output.FullName, file.Name);

                                        try
                                        {
                                            if(!options.DryRun && !options.Overwrite && File.Exists(outputPath))
                                            {
                                                if(options.Verbose) progress.WriteLine($"skip (exists): {file.Name}");
                                                progress.Advance();

                                                return ValueTask.CompletedTask;
                                            }

                                            var r = InlayPipeline.ProcessFile(file.FullName, outputPath, options);
                                            results.Add(r);
                                            var n = Interlocked.Increment(ref done);

                                            var note = r.Detected
                                                           ? $"skew {r.Skew,6:0.00}°, orient {r.Orientation}°, {r.Width}x{r.Height}"
                                                           : "NO INLAY DETECTED (copied untouched)";

                                            progress.WriteLine(options.Verbose
                                                                   ? $"[{n}/{files.Length}] {file.Name}: {note} conf={r.Confidence:0.00}"
                                                                   : $"[{n}/{files.Length}] {file.Name}: {note}");
                                        }
                                        catch(Exception ex) when(ex is not OperationCanceledException)
                                        {
                                            failures.Add((file.Name, ex.Message));
                                            progress.WriteLine($"FAILED {file.Name}: {ex.Message}", error: true);
                                        }

                                        progress.Advance();

                                        return ValueTask.CompletedTask;
                                    });

        progress.Dispose(); // clear the bar before the summary

        if(options.Report is {} report)
        {
            var lines = new List<string>
            {
                "file,skew_deg,orientation_deg,confidence,width,height,detected"
            };

            lines.AddRange(results.OrderBy(r => r.File, StringComparer.OrdinalIgnoreCase)
                                  .Select(r => string.Create(CultureInfo.InvariantCulture,
                                                             $"{r.File},{r.Skew:0.00},{r.Orientation},{r.Confidence:0.00},{r.Width},{r.Height},{r.Detected}")));

            lines.AddRange(failures.Select(f => $"{f.File},,,,,,error: {f.Error.Replace(',', ';')}"));
            await File.WriteAllLinesAsync(report.FullName, lines, ct);
            Console.WriteLine($"Report written to {report.FullName}");
        }

        Console.WriteLine($"Done: {results.Count} processed, {failures.Count} failed" +
                          (options.DryRun ? " (dry run, nothing written)" : ""));

        return failures.IsEmpty ? 0 : 2;
    }
}
