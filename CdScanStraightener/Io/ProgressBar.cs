using System.Diagnostics;

namespace CdScanStraightener.Io;

/// <summary>
/// In-place console progress bar with percentage and ETA, drawn on stderr so redirected
/// stdout stays clean. When stderr is not a terminal it stays silent — the per-file
/// lines already provide progress in logs. All console output must go through
/// <see cref="WriteLine"/> so the bar can be cleared and redrawn beneath it.
/// </summary>
public sealed class ProgressBar(int total) : IDisposable
{
    private readonly Stopwatch _watch   = Stopwatch.StartNew();
    private readonly bool      _enabled = !Console.IsErrorRedirected;
    private readonly Lock      _lock    = new();
    private          int       _done;

    public void Advance()
    {
        lock(_lock)
        {
            _done++;
            Redraw();
        }
    }

    /// <summary>Prints a message above the bar (stdout), keeping the bar at the bottom.</summary>
    public void WriteLine(string message, bool error = false)
    {
        lock(_lock)
        {
            Clear();
            (error ? Console.Error : Console.Out).WriteLine(message);
            Redraw();
        }
    }

    private void Redraw()
    {
        if(!_enabled) return;
        const int width  = 30;
        var       frac   = total == 0 ? 1.0 : (double)_done / total;
        var       filled = (int)(frac * width);
        var       eta    = "";

        if(_done > 0 && _done < total)
        {
            var remaining = TimeSpan.FromSeconds(_watch.Elapsed.TotalSeconds / _done * (total - _done));

            eta = remaining.TotalHours >= 1
                      ? $" ETA {(int)remaining.TotalHours}h{remaining.Minutes:00}m"
                      : $" ETA {(int)remaining.TotalMinutes}m{remaining.Seconds:00}s";
        }

        Console.Error
               .Write($"\r[{new string('█', filled)}{new string('░', width - filled)}] {_done}/{total} ({frac:P0}){eta}   ");
    }

    private void Clear()
    {
        if(!_enabled) return;
        Console.Error.Write("\r" + new string(' ', Console.IsErrorRedirected ? 0 : SafeWidth()) + "\r");
    }

    private static int SafeWidth()
    {
        try
        {
            return Math.Max(20, Console.WindowWidth - 1);
        }
        catch
        {
            return 80;
        }
    }

    public void Dispose()
    {
        lock(_lock)
        {
            Clear();
        }
    }
}