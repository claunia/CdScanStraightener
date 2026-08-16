using System.Collections.Concurrent;
using System.Globalization;
using CdScanStraightener.Cli;
using CdScanStraightener.Common;
using CdScanStraightener.Pipeline;

namespace CdScanStraightener.Io;

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

        var       results  = new ConcurrentBag<AngleResult>();
        var       failures = new ConcurrentBag<(string File, string Error)>();
        var       done     = 0;
        using var progress = new ProgressBar(files.Length);

        await Parallel.ForEachAsync(files,
                                    new ParallelOptions
                                    {
                                        MaxDegreeOfParallelism = Environment.ProcessorCount,
                                        CancellationToken      = ct
                                    },
                                    (file, token) =>
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

                                            var result =
                                                StraightenPipeline.ProcessFile(file.FullName, outputPath, options);

                                            results.Add(result);
                                            var n = Interlocked.Increment(ref done);

                                            var note = result.Applied
                                                           ? $"{result.Angle,7:0.00}°"
                                                           : $"{result.Angle,7:0.00}° NOT APPLIED (confidence {result.Confidence:0.00} below threshold)";

                                            if(options.Verbose)
                                                progress
                                                   .WriteLine($"[{n}/{files.Length}] {file.Name}: {note} conf={result.Confidence:0.00} ({result.Method})");
                                            else
                                                progress.WriteLine($"[{n}/{files.Length}] {file.Name}: {note}");
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
                "file,angle_deg,confidence,applied,method,disc_cx,disc_cy,disc_radius"
            };

            lines.AddRange(results.OrderBy(r => r.File, StringComparer.OrdinalIgnoreCase)
                                  .Select(r => string.Create(CultureInfo.InvariantCulture,
                                                             $"{r.File},{r.Angle:0.00},{r.Confidence:0.00},{r.Applied},{r.Method},{r.DiscCenter.X:0.0},{r.DiscCenter.Y:0.0},{r.DiscRadius:0.0}")));

            lines.AddRange(failures.Select(f => $"{f.File},,,,error: {f.Error.Replace(',', ';')},,,"));
            await File.WriteAllLinesAsync(report.FullName, lines, ct);
            Console.WriteLine($"Report written to {report.FullName}");
        }

        Console.WriteLine($"Done: {results.Count} processed, {failures.Count} failed" +
                          (options.DryRun ? " (dry run, nothing written)" : ""));

        return failures.IsEmpty ? 0 : 2;
    }
}