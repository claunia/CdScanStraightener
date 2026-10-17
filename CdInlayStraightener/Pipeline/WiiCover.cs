using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

/// <summary>A Wii keep-case cover: the upright crop, the rotation that straightened it, and the
/// quarter turns (counterclockwise) applied on top.</summary>
public sealed record WiiCoverResult(Mat Image, double Skew, int Quarters) : IDisposable
{
    public void Dispose() => Image.Dispose();
}

/// <summary>
/// Wii keep-case covers follow one printed layout, which pins their geometry where the paper
/// edges cannot: the cover is 273 mm long and so is the scanner bed, so it lies against the
/// bed's edge, white paper on a white lid. Two landmarks give the whole frame:
///
/// - the grey "Wii" wordmark at the top of the spine fixes orientation and position;
/// - the colored right triangle at the bottom of the spine — its top leg on the front fold,
///   its right leg on the paper's bottom edge — is 2075 px along the spine from the logo and
///   30 px above it (at 300 dpi). The vector between the two gives the rotation and scale.
///
/// From there the model places the remaining edges: the top end 2156 px before the corner,
/// the front edge 1534 px before the fold, the back edge 1690 px after it. Cover trims vary by
/// a few millimetres, so an edge moves inward when everything from it out to the scan border
/// is plain scanner lid; edges that coincide with the scan border keep the model's position.
/// </summary>
public static class WiiCover
{
    private const double Dpi           = 300;
    private const double Along         = 2075;
    private const double Above         = -30;
    private const double TopEnd        = 2156;
    private const double FrontEdge     = 1534;
    private const double BackEdge      = 1690;
    private const double WorkDpi       = 75;
    private const double MinLogoScore  = 0.8;

    private static readonly Lazy<Mat?> Template = new(() =>
    {
        var dirs = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Templates"), Path.Combine(Environment.CurrentDirectory, "Templates")
        };

        var file = dirs.Select(d => Path.Combine(d, "wii-spine.png")).FirstOrDefault(File.Exists);

        if(file is null) return null;

        var img = Cv2.ImRead(file, ImreadModes.Grayscale);

        return img.Empty() ? null : img;
    });

    /// <summary>Detects and crops a Wii cover, or null when the scan is not one.</summary>
    public static WiiCoverResult? Detect(Mat src, double? dpi)
    {
        if(Template.Value is not {} template) return null;

        var f = (dpi ?? Dpi) / Dpi;

        using var gray = new Mat();
        Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

        // 1. Spine logo at a coarse scale, in all four orientations.
        var ws = WorkDpi / (dpi ?? Dpi);

        using var small = new Mat();
        Cv2.Resize(gray, small, default, ws, ws, InterpolationFlags.Area);

        var (score, quarters) = FindLogo(small, template, WorkDpi / Dpi);

        if(score < MinLogoScore) return null;

        // 2. Normalize losslessly: spine logo upright, front panel above the spine.
        using var normalized = RotateClockwise(src, quarters);
        using var normGray   = RotateClockwise(gray, quarters);

        if(LocateLogo(normGray, template, f, ws) is not {} logo) return null;

        // 3. The triangle corner, where the model expects it.
        if(Corner(normalized, logo.X + Along * f, logo.Y + Above * f) is not {} corner) return null;

        var dx    = corner.X - logo.X;
        var dy    = corner.Y - logo.Y;
        var theta = Math.Atan2(dy, dx) - Math.Atan2(Above, Along);

        // Measured against modeled landmark distance: the scanner's true scale relative to its DPI.
        var k = Math.Sqrt(dx * dx + dy * dy) / (Math.Sqrt(Along * Along + Above * Above) * f);

        // Implausible scale means the corner is something else.
        if(Math.Abs(k - 1) > 0.02) return null;

        var scale = f * k;

        // 4. Deskew about the logo.
        var skew = theta * 180 / Math.PI;
        int w    = normalized.Width, h = normalized.Height;

        using var m       = Cv2.GetRotationMatrix2D(new Point2f((float)logo.X, (float)logo.Y), skew, 1.0);
        using var rotated = new Mat();
        using var valid   = new Mat();
        using var ones    = new Mat(normalized.Size(), MatType.CV_8U, Scalar.All(255));

        Cv2.WarpAffine(normalized, rotated, m, normalized.Size(), InterpolationFlags.Lanczos4, BorderTypes.Constant,
                       Scalar.All(0));

        Cv2.WarpAffine(ones, valid, m, normalized.Size(), InterpolationFlags.Nearest, BorderTypes.Constant,
                       Scalar.All(0));

        m.GetArray(out double[] mm);
        var cx = mm[0] * corner.X + mm[1] * corner.Y + mm[2];
        var cy = mm[3] * corner.X + mm[4] * corner.Y + mm[5];

        // 5. Model box, then lid evidence.
        double l = cx - TopEnd * scale, r = cx, t = cy - FrontEdge * scale, b = cy + BackEdge * scale;

        using var rotGray = new Mat();
        Cv2.CvtColor(rotated, rotGray, ColorConversionCodes.BGR2GRAY);
        rotGray.GetArray(out byte[] px);
        valid.GetArray(out byte[] ok);

        var lid = LidLevel(px, ok, w, h, (int)l - 30);

        var rows = new Profile(px, ok, w, h, true, (int)(l + 150 * scale), (int)(r - 150 * scale));

        if(rows.EdgeFrom(0, (int)(t + 80 * scale), 1, lid) is {} front && front > t - 80 * scale)
            t = Math.Max(t, front);

        if(rows.EdgeFrom(h - 1, (int)(b - 80 * scale), -1, lid) is {} back && back < b + 80 * scale)
            b = Math.Min(b, back + 1);

        var cols = new Profile(px, ok, w, h, false, (int)(Math.Max(t, 0) + 150 * scale),
                               (int)(Math.Min(b, h) - 150 * scale));

        if(cols.EdgeFrom(0, (int)(l + 40 * scale), 1, lid) is {} top && top > l - 40 * scale) l = Math.Max(l, top);

        // 6. Only what the scanner saw: clip to the scan, then pull in sides that still touch the
        // wedges the rotation brought in from outside it.
        int x0 = Math.Max(0, (int)Math.Ceiling(l)), x1 = Math.Min(w, (int)Math.Floor(r));
        int y0 = Math.Max(0, (int)Math.Ceiling(t)), y1 = Math.Min(h, (int)Math.Floor(b));

        bool RowValid(int y) => Enumerable.Range(x0, x1 - x0).All(x => ok[y * w + x] != 0);
        bool ColValid(int x) => Enumerable.Range(y0, y1 - y0).All(y => ok[y * w + x] != 0);

        for(var i = 0; i < 100 && x1 - x0 > 100 && y1 - y0 > 100; i++)
        {
            var moved = false;

            if(!RowValid(y0)) { y0++; moved = true; }
            if(!RowValid(y1 - 1)) { y1--; moved = true; }
            if(!ColValid(x0)) { x0++; moved = true; }
            if(!ColValid(x1 - 1)) { x1--; moved = true; }

            if(!moved) break;
        }

        if(x1 - x0 < 1000 * f || y1 - y0 < 1000 * f) return null;

        // 7. Upright: top end up, front on the right.
        using var crop    = rotated[new Rect(x0, y0, x1 - x0, y1 - y0)];
        var       upright = new Mat();
        Cv2.Rotate(crop, upright, RotateFlags.Rotate90Clockwise);

        // Clockwise turns applied: `quarters` to normalize, one to stand it up.
        return new WiiCoverResult(upright, skew, (4 - (quarters + 1) % 4) % 4);
    }

    private static Mat RotateClockwise(Mat src, int quarters)
    {
        var dst = new Mat();

        switch(quarters % 4)
        {
            case 0:
                src.CopyTo(dst);

                break;
            case 1:
                Cv2.Rotate(src, dst, RotateFlags.Rotate90Clockwise);

                break;
            case 2:
                Cv2.Rotate(src, dst, RotateFlags.Rotate180);

                break;
            default:
                Cv2.Rotate(src, dst, RotateFlags.Rotate90Counterclockwise);

                break;
        }

        return dst;
    }

    /// <summary>
    /// Best normalized correlation of the spine logo over the four orientations and a small scale
    /// range; also how many clockwise quarter turns make the scan's logo upright.
    /// </summary>
    private static (double Score, int Quarters) FindLogo(Mat small, Mat template, double templateScale)
    {
        var best = (Score: -1.0, Quarters: 0);

        foreach(var s in new[] { 0.92, 1.0, 1.08 })
        {
            using var t0 = new Mat();
            Cv2.Resize(template, t0, default, templateScale * s, templateScale * s, InterpolationFlags.Area);

            for(var q = 0; q < 4; q++)
            {
                // The logo appears turned q quarters counterclockwise; q clockwise turns undo it.
                using var t = RotateClockwise(t0, (4 - q) % 4);

                if(t.Width >= small.Width || t.Height >= small.Height) continue;

                using var result = new Mat();
                Cv2.MatchTemplate(small, t, result, TemplateMatchModes.CCoeffNormed);
                Cv2.MinMaxLoc(result, out double _, out double max);

                if(max > best.Score) best = (max, q);
            }
        }

        return best;
    }

    /// <summary>Top-left of the upright spine logo at full resolution.</summary>
    private static Point2d? LocateLogo(Mat gray, Mat template, double f, double ws)
    {
        using var t = new Mat();
        Cv2.Resize(template, t, default, f, f, InterpolationFlags.Area);

        using var small = new Mat();
        Cv2.Resize(gray, small, default, ws, ws, InterpolationFlags.Area);

        using var ts = new Mat();
        Cv2.Resize(t, ts, default, ws, ws, InterpolationFlags.Area);

        using var coarse = new Mat();
        Cv2.MatchTemplate(small, ts, coarse, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(coarse, out _, out var score, out _, out var at);

        if(score < MinLogoScore) return null;

        // Refine in a window at full resolution.
        var pad = (int)(60 * f);
        var x0  = Math.Max(0, (int)(at.X / ws) - pad);
        var y0  = Math.Max(0, (int)(at.Y / ws) - pad);
        var x1  = Math.Min(gray.Width, (int)(at.X / ws) + t.Width + pad);
        var y1  = Math.Min(gray.Height, (int)(at.Y / ws) + t.Height + pad);

        using var window = gray[new Rect(x0, y0, x1 - x0, y1 - y0)];
        using var fine   = new Mat();
        Cv2.MatchTemplate(window, t, fine, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(fine, out _, out _, out _, out var loc);

        return new Point2d(x0 + loc.X, y0 + loc.Y);
    }

    /// <summary>
    /// The triangle's right-angle corner near (px, py): sample its color just inside, grow the
    /// region of that (smoothed — the print is halftoned) color, and read its top leg and right
    /// leg off the region's outline.
    /// </summary>
    private static Point2d? Corner(Mat img, double px, double py)
    {
        int w = img.Width, h = img.Height;
        int sx = (int)(px - 20), sy = (int)(py + 60);

        var x0 = Math.Max(0, (int)px - 260);
        var x1 = Math.Min(w, (int)px + 60);
        var y0 = Math.Max(0, (int)py - 120);
        var y1 = Math.Min(h, (int)py + 260);

        if(sx < x0 || sx >= x1 || sy < y0 || sy >= y1 || x1 - x0 < 100 || y1 - y0 < 100) return null;

        using var window = img[new Rect(x0, y0, x1 - x0, y1 - y0)];
        using var blur   = new Mat();
        Cv2.Blur(window, blur, new Size(7, 7));

        using var lab8 = new Mat();
        Cv2.CvtColor(blur, lab8, ColorConversionCodes.BGR2Lab);

        using var lab = new Mat();
        lab8.ConvertTo(lab, MatType.CV_32FC3);

        var reference = lab.At<Vec3f>(sy - y0, sx - x0);

        // OpenCV 8-bit Lab: paper white is about (250, 128, 128). Sampling paper means no triangle.
        var fromPaper = Math.Sqrt(Math.Pow(reference.Item0 - 250, 2) + Math.Pow(reference.Item1 - 128, 2) +
                                  Math.Pow(reference.Item2 - 128, 2));

        if(fromPaper < 15) return null;

        using var diff = new Mat();
        Cv2.Subtract(lab, new Scalar(reference.Item0, reference.Item1, reference.Item2), diff);
        Cv2.Multiply(diff, diff, diff);

        var channels = Cv2.Split(diff);
        using var dist = new Mat();
        Cv2.Add(channels[0], channels[1], dist);
        Cv2.Add(dist, channels[2], dist);
        foreach(var c in channels) c.Dispose();

        using var mask = new Mat();
        Cv2.Threshold(dist, mask, 18 * 18, 255, ThresholdTypes.BinaryInv);
        mask.ConvertTo(mask, MatType.CV_8U);

        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);

        using var labels = new Mat();
        Cv2.ConnectedComponents(mask, labels);

        var id = labels.At<int>(sy - y0, sx - x0);

        if(id == 0) return null;

        labels.GetArray(out int[] lbl);
        int ww = x1 - x0, wh = y1 - y0;

        var right = -1;

        for(var i = 0; i < lbl.Length; i++)
            if(lbl[i] == id)
                right = Math.Max(right, i % ww);

        if(right < 40) return null;

        // Top leg: top row in the columns just left of the right leg.
        var tops = new List<int>();

        for(var c = right - 30; c < right - 5; c++)
            for(var y = 0; y < wh; y++)
                if(lbl[y * ww + c] == id)
                {
                    tops.Add(y);

                    break;
                }

        if(tops.Count == 0) return null;

        tops.Sort();
        var top = tops[tops.Count / 2];

        // Right leg: rightmost column in the rows just below the top leg.
        var rights = new List<int>();

        for(var y = top + 5; y < Math.Min(wh, top + 60); y++)
            for(var c = ww - 1; c >= 0; c--)
                if(lbl[y * ww + c] == id)
                {
                    rights.Add(c);

                    break;
                }

        if(rights.Count == 0) return null;

        rights.Sort();

        return new Point2d(x0 + rights[rights.Count / 2], y0 + top);
    }

    /// <summary>Median of the scanned pixels left of <paramref name="limit"/>: the lid beyond the cover's top end.</summary>
    private static double LidLevel(byte[] px, byte[] ok, int w, int h, int limit)
    {
        limit = Math.Clamp(limit, 5, w);
        var values = new List<byte>();

        for(var y = 0; y < h; y += 3)
            for(var x = 0; x < limit; x += 3)
                if(ok[y * w + x] != 0)
                    values.Add(px[y * w + x]);

        if(values.Count < 1000) return 250;

        values.Sort();

        return values[values.Count / 2];
    }

    /// <summary>Per-row (or per-column) median and spread of scanned pixels over a band, computed on demand.</summary>
    private sealed class Profile(byte[] px, byte[] ok, int w, int h, bool rows, int from, int to)
    {
        private readonly Dictionary<int, (double Median, double Sd, bool Valid)> _cache = new();

        private (double Median, double Sd, bool Valid) At(int i)
        {
            if(_cache.TryGetValue(i, out var hit)) return hit;

            var lo = Math.Clamp(from, 0, rows ? w : h);
            var hi = Math.Clamp(to, 0, rows ? w : h);
            var v  = new List<byte>(Math.Max(0, hi - lo));

            for(var j = lo; j < hi; j++)
            {
                var idx = rows ? i * w + j : j * w + i;

                if(ok[idx] != 0) v.Add(px[idx]);
            }

            if(v.Count < 0.5 * (hi - lo) || v.Count == 0) return _cache[i] = (0, 0, false);

            v.Sort();
            var mean = v.Average(b => (double)b);
            var sd   = Math.Sqrt(v.Average(b => (b - mean) * (b - mean)));

            return _cache[i] = (v[v.Count / 2], sd, true);
        }

        /// <summary>
        /// Walks from the scan border inward and returns the first position where three consecutive
        /// lines are no longer plain lid — the paper's edge — or null if it never leaves the lid.
        /// </summary>
        public int? EdgeFrom(int start, int stop, int step, double lid)
        {
            var n   = rows ? h : w;
            var run = 0;

            for(var i = start; (i - stop) * step < 0 && i >= 0 && i < n; i += step)
            {
                var (median, sd, valid) = At(i);

                if(!valid) continue;

                var isLid = Math.Abs(median - lid) <= 4 && sd < 5;
                run = isLid ? 0 : run + 1;

                if(run == 3) return i - 2 * step;
            }

            return null;
        }
    }
}
