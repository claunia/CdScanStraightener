using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

/// <summary>
/// A detected inlay: deskewing the scan by <paramref name="Skew"/> degrees about its center
/// (counterclockwise, as Cv2.GetRotationMatrix2D takes it) makes the inlay the axis-aligned
/// <paramref name="Box"/>, in full-resolution pixels of that deskewed scan.
/// </summary>
public sealed record Inlay(double Skew, Rect Box, Scalar Background);

/// <summary>
/// Finds the inlay — paper, a box side, a cartridge — by its outline rather than its color,
/// so it works when the item matches the scanner lid: black prints on the black lid, white
/// paper on the white lid. Physical edges are long, straight and sharp, and the outermost
/// one on each side is the item's. Where an edge leaves no trace at all (white paper, white
/// lid, the side away from the light) the side falls back to the printed content, or — given
/// the scan's DPI — to a standard inlay size measured from the opposite edge.
/// </summary>
public static class InlayDetector
{
    /// <summary>Detection runs on a copy scaled down to this longest side.</summary>
    private const int WorkSide = 1200;

    /// <summary>Common physical inlay formats in millimetres (width × height, either orientation).</summary>
    private static readonly (double W, double H)[] FormatsMm =
    [
        (120, 120), // CD front inlay / booklet
        (240, 120), // CD booklet, unfolded
        (151, 118), // CD tray inlay with spines
        (138, 118), // CD tray inlay without spines
        (273, 183), // DVD / Wii / PS2 keep-case cover
    ];

    /// <summary>Median color of the outer ~1% frame — the scanner background.</summary>
    public static Scalar DetectBackground(Mat src)
    {
        var t = Math.Max(4, Math.Min(src.Rows, src.Cols) / 100);

        var strips = new[]
        {
            new Rect(0, 0, src.Cols, t),
            new Rect(0, src.Rows - t, src.Cols, t),
            new Rect(0, 0, t, src.Rows),
            new Rect(src.Cols - t, 0, t, src.Rows),
        };

        var b = new List<byte>();
        var g = new List<byte>();
        var r = new List<byte>();

        foreach(var strip in strips)
        {
            using var roi = src[strip];
            // Sample sparsely — exact medians over megapixel strips are unnecessary.
            for(var y = 0; y < roi.Rows; y += 3)
                for(var x = 0; x < roi.Cols; x += 7)
                {
                    var px = roi.At<Vec3b>(y, x);
                    b.Add(px.Item0);
                    g.Add(px.Item1);
                    r.Add(px.Item2);
                }
        }

        b.Sort();
        g.Sort();
        r.Sort();

        return new Scalar(b[b.Count / 2], g[g.Count / 2], r[r.Count / 2]);
    }

    /// <summary>
    /// Detects the inlay, or null if the scan shows nothing but background. <paramref name="dpi"/>
    /// enables placing invisible edges at standard inlay sizes.
    /// </summary>
    public static Inlay? Detect(Mat src, double? dpi = null)
    {
        var scale = Math.Min(1.0, (double)WorkSide / Math.Max(src.Width, src.Height));

        using var small = new Mat();

        if(scale < 1.0)
            Cv2.Resize(src, small, default, scale, scale, InterpolationFlags.Area);
        else
            src.CopyTo(small);

        var background = DetectBackground(small);
        var workDpi    = dpi * scale;

        var best = Analyze(small, background, null, workDpi);

        if(best is null) return null;

        // The first skew comes from all long lines in the scan; refine it from the edge pixels
        // along the four detected sides only, where artwork cannot vote.
        var residual = best.SideResidual();

        if(Math.Abs(residual) > 0.1)
        {
            foreach(var candidate in new[] { best.Skew + residual, best.Skew - residual })
            {
                var retry = Analyze(small, background, candidate, workDpi);

                if(retry is not null && Math.Abs(retry.SideResidual()) < Math.Abs(residual))
                {
                    best.Dispose();
                    best = retry;

                    break;
                }

                retry?.Dispose();
            }
        }

        using(best)
        {
            if(Environment.GetEnvironmentVariable("CDSCAN_ARC_DUMP") is {} dump) best.Dump(dump);

            var (l, t, r, b) = best.Box;

            if((r - l + 1) * (b - t + 1) < 0.01 * small.Width * small.Height) return null;

            // Back to full resolution; the deskew is about the image center at both scales. The
            // edge pixels themselves are uncertain by a working pixel, so drop them.
            var x0 = (int)Math.Ceiling((l + 1) / scale);
            var y0 = (int)Math.Ceiling((t + 1) / scale);
            var x1 = Math.Min(src.Width, (int)Math.Floor(r / scale));
            var y1 = Math.Min(src.Height, (int)Math.Floor(b / scale));

            return new Inlay(best.Skew, new Rect(x0, y0, x1 - x0, y1 - y0), background);
        }
    }

    /// <summary>
    /// One detection pass at a given skew (estimated when null). Returns null when nothing
    /// but background is found.
    /// </summary>
    private static Analysis? Analyze(Mat small, Scalar background, double? skew, double? dpi)
    {
        int w = small.Width, h = small.Height;

        using var img = new Mat();
        small.ConvertTo(img, MatType.CV_32FC3);

        var threshold = NoiseThreshold(img);

        using var ex0 = SharpEdges(img, true, threshold);
        using var ey0 = SharpEdges(img, false, threshold);

        if(skew is null)
        {
            using var both = new Mat();
            Cv2.BitwiseOr(ex0, ey0, both);
            skew = SkewFromLines(both);
        }

        DropScannerLines(ex0, ey0, skew.Value);

        using var m       = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), skew.Value, 1.0);
        var       rotated = new Mat();
        var       ex      = new Mat();
        var       ey      = new Mat();

        Cv2.WarpAffine(small, rotated, m, small.Size(), InterpolationFlags.Linear, BorderTypes.Replicate);
        Cv2.WarpAffine(ex0, ex, m, small.Size(), InterpolationFlags.Nearest);
        Cv2.WarpAffine(ey0, ey, m, small.Size(), InterpolationFlags.Nearest);

        var analysis = new Analysis(skew.Value, rotated, ex, ey);

        var content = ContentBox(rotated, background);

        if(content is null)
        {
            analysis.Dispose();

            return null;
        }

        analysis.Content = content.Value;
        analysis.FindSides();

        var lightLid = (background.Val0 + background.Val1 + background.Val2) / 3;

        // Cast shadows and penumbras only show on a light lid.
        if(lightLid > 128) analysis.TrimShadows();

        // White paper on a white lid: invisible sides go to a standard size if one fits.
        if(lightLid > 200 && dpi is {} d && !analysis.Certain.All(c => c)) analysis.SnapToFormat(d);

        return analysis;
    }

    /// <summary>Edge-strength threshold: four times the 75th percentile of gradients in the scanner frame, at least 10.</summary>
    private static double NoiseThreshold(Mat img)
    {
        using var gx = MaxAbsGradient(img, 1, 0);

        var t      = Math.Max(4, Math.Min(img.Rows, img.Cols) / 100);
        var values = new List<float>();

        foreach(var strip in new[]
                {
                    new Rect(0, 0, gx.Cols, t), new Rect(0, gx.Rows - t, gx.Cols, t), new Rect(0, 0, t, gx.Rows),
                    new Rect(gx.Cols - t, 0, t, gx.Rows)
                })
        {
            using var roi = gx[strip].Clone();
            roi.GetArray(out float[] data);
            values.AddRange(data);
        }

        values.Sort();

        return Math.Max(10.0, 4 * values[(int)(0.75 * (values.Count - 1))]);
    }

    /// <summary>Per-pixel maximum over the color channels of |Sobel|.</summary>
    private static Mat MaxAbsGradient(Mat img, int dx, int dy)
    {
        using var sobel = new Mat();
        Cv2.Sobel(img, sobel, MatType.CV_32F, dx, dy, 3);

        var channels = Cv2.Split(sobel);
        var result   = new Mat();

        try
        {
            foreach(var c in channels) Cv2.Absdiff(c, Scalar.All(0), c);

            Cv2.Max(channels[0], channels[1], result);
            Cv2.Max(result, channels[2], result);
        }
        finally
        {
            foreach(var c in channels) c.Dispose();
        }

        return result;
    }

    /// <summary>
    /// Binary (0/255) map of sharp edges across the given direction: strong gradient that also
    /// stands out from its own neighborhood, which keeps steps and thin lines but rejects soft
    /// ramps such as shadows and lid shading.
    /// </summary>
    private static Mat SharpEdges(Mat img, bool vertical, double threshold)
    {
        using var s     = vertical ? MaxAbsGradient(img, 1, 0) : MaxAbsGradient(img, 0, 1);
        using var local = new Mat();
        Cv2.Blur(s, local, vertical ? new Size(31, 1) : new Size(1, 31));
        Cv2.Multiply(local, Scalar.All(2.5), local);

        using var standsOut = new Mat();
        Cv2.Compare(s, local, standsOut, CmpTypes.GT);

        using var strong = new Mat();
        Cv2.Threshold(s, strong, threshold, 255, ThresholdTypes.Binary);
        strong.ConvertTo(strong, MatType.CV_8U);

        var edges = new Mat();
        Cv2.BitwiseAnd(standsOut, strong, edges);

        return edges;
    }

    /// <summary>Dominant tilt (degrees, ±20) of the long straight segments in an edge map.</summary>
    private static double SkewFromLines(Mat edges)
    {
        var minLength = (int)(0.15 * Math.Min(edges.Rows, edges.Cols));
        var lines     = Cv2.HoughLinesP(edges, 1, Math.PI / 720, minLength / 2, minLength, 3);

        const int    bins  = 160;
        const double range = 20;
        var          hist  = new double[bins];
        var          found = new List<(double Angle, double Length)>();

        foreach(var l in lines)
        {
            var angle = Math.Atan2(l.P2.Y - l.P1.Y, l.P2.X - l.P1.X) * 180 / Math.PI;
            angle = ((angle + 45) % 90 + 90) % 90 - 45;

            if(Math.Abs(angle) >= range) continue;

            var length = Math.Sqrt(Math.Pow(l.P2.X - l.P1.X, 2) + Math.Pow(l.P2.Y - l.P1.Y, 2));
            found.Add((angle, length));
            hist[Math.Min(bins - 1, (int)((angle + range) / (2 * range) * bins))] += length;
        }

        if(found.Count == 0) return 0;

        // Length-weighted mean around the length-weighted mode.
        var peak  = Array.IndexOf(hist, hist.Max());
        var width = 2 * range / bins;
        var lo    = -range + (peak - 1) * width;
        var hi    = -range + (peak + 2) * width;
        var near  = found.Where(f => f.Angle >= lo && f.Angle <= hi).ToArray();

        return near.Sum(f => f.Angle * f.Length) / near.Sum(f => f.Length);
    }

    /// <summary>
    /// Scanner bed and lid edges are perfectly axis-aligned and long. A line from border to
    /// border is never an item edge, and once the item is measurably tilted its own edges cannot
    /// form long axis-aligned runs either — so erase such lines from the unrotated edge maps.
    /// </summary>
    private static void DropScannerLines(Mat ex, Mat ey, double skew)
    {
        int w = ex.Cols, h = ex.Rows;
        var f = Math.Abs(skew) >= 0.5 ? 0.75 : 0.97;

        var cols = LongestRuns(ex, true, w, h);

        for(var x = 0; x < w; x++)
        {
            if(cols[x] < f * h) continue;

            using var roi = ex[new Rect(Math.Max(0, x - 1), 0, Math.Min(w, x + 2) - Math.Max(0, x - 1), h)];
            roi.SetTo(Scalar.All(0));
        }

        var rows = LongestRuns(ey, false, w, h);

        for(var y = 0; y < h; y++)
        {
            if(rows[y] < f * w) continue;

            using var roi = ey[new Rect(0, Math.Max(0, y - 1), w, Math.Min(h, y + 2) - Math.Max(0, y - 1))];
            roi.SetTo(Scalar.All(0));
        }
    }

    /// <summary>
    /// Longest run of edge pixels along each column (vertical) or row, after tolerating 1 px of
    /// wobble across and gaps of 1% of the scan along the line.
    /// </summary>
    private static int[] LongestRuns(Mat edges, bool vertical, int w, int h)
    {
        var gap = Math.Max(3, (int)(0.01 * Math.Max(w, h)));

        using var joined = new Mat();
        using var across = Cv2.GetStructuringElement(MorphShapes.Rect, vertical ? new Size(3, 1) : new Size(1, 3));
        using var along  = Cv2.GetStructuringElement(MorphShapes.Rect, vertical ? new Size(1, gap) : new Size(gap, 1));
        Cv2.Dilate(edges, joined, across);
        Cv2.MorphologyEx(joined, joined, MorphTypes.Close, along);

        joined.GetArray(out byte[] px);

        var n    = vertical ? w : h;
        var len  = vertical ? h : w;
        var best = new int[n];

        for(var i = 0; i < n; i++)
        {
            var run = 0;

            for(var j = 0; j < len; j++)
            {
                var on = vertical ? px[j * w + i] != 0 : px[i * w + j] != 0;
                run = on ? run + 1 : 0;

                if(run > best[i]) best[i] = run;
            }
        }

        return best;
    }

    /// <summary>Bounding box (inclusive L, T, R, B) of everything clearly different from the background, or null.</summary>
    private static (int L, int T, int R, int B)? ContentBox(Mat img, Scalar background)
    {
        using var blurred = new Mat();
        Cv2.GaussianBlur(img, blurred, new Size(5, 5), 0);

        using var diff = new Mat();
        Cv2.Absdiff(blurred, background, diff);
        Cv2.CvtColor(diff, diff, ColorConversionCodes.BGR2GRAY);

        using var mask = new Mat();
        Cv2.Threshold(diff, mask, 40, 255, ThresholdTypes.Binary);

        // Drop dust specks.
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(5, 5));
        Cv2.MorphologyEx(mask, mask, MorphTypes.Open, kernel);

        DropBedStrips(mask);

        using var points = new Mat();
        Cv2.FindNonZero(mask, points);

        if(points.Empty()) return null;

        var bb = Cv2.BoundingRect(points);

        return (bb.Left, bb.Top, bb.Right - 1, bb.Bottom - 1);
    }

    /// <summary>
    /// A scan that reaches past the lid shows the dark scanner bed as a strip along a border,
    /// running the full length of the scan. It is not content, and counting it puts the content
    /// box on the scan border, where no inlay side can be looked for. A strip is the band of
    /// lines from the border up to the first background-only line, within 3% of the scan, of
    /// which at least one is covered for half its length; an inlay running off the scan has no
    /// background gap and is kept.
    /// </summary>
    private static void DropBedStrips(Mat mask)
    {
        int w = mask.Cols, h = mask.Rows;

        mask.GetArray(out byte[] px);

        var colCover = new int[w];
        var rowCover = new int[h];

        for(var y = 0; y < h; y++)
            for(var x = 0; x < w; x++)
            {
                if(px[y * w + x] == 0) continue;

                colCover[x]++;
                rowCover[y]++;
            }

        int? StripEnd(int[] cover, int length, int from, int step)
        {
            var reach = Math.Max(4, (int)(0.03 * cover.Length));
            var long_ = false;

            for(var k = 0; k < reach; k++)
            {
                var i = from + step * k;

                if(cover[i] <= 0.02 * length) return long_ ? i : null;

                if(cover[i] >= 0.5 * length) long_ = true;
            }

            return null;
        }

        if(StripEnd(colCover, h, 0, 1) is {} l) mask[new Rect(0, 0, l, h)].SetTo(Scalar.All(0));

        if(StripEnd(colCover, h, w - 1, -1) is {} r) mask[new Rect(r + 1, 0, w - r - 1, h)].SetTo(Scalar.All(0));

        if(StripEnd(rowCover, w, 0, 1) is {} t) mask[new Rect(0, 0, w, t)].SetTo(Scalar.All(0));

        if(StripEnd(rowCover, w, h - 1, -1) is {} b) mask[new Rect(0, b + 1, w, h - b - 1)].SetTo(Scalar.All(0));
    }

    /// <summary>One detection pass on the deskewed working-scale scan.</summary>
    private sealed class Analysis(double skew, Mat rotated, Mat ex, Mat ey) : IDisposable
    {
        public double Skew { get; } = skew;

        /// <summary>Inclusive L, T, R, B of the inlay in the deskewed working-scale scan.</summary>
        public (int L, int T, int R, int B) Box { get; private set; }

        public (int L, int T, int R, int B) Content { get; set; }

        /// <summary>Per side (L, T, R, B): whether a straight line was found there at all.</summary>
        public bool[] Lines { get; } = new bool[4];

        /// <summary>Per side: whether that line is evidence of the physical edge rather than of printed content.</summary>
        public bool[] Certain { get; } = new bool[4];

        private int W => rotated.Width;
        private int H => rotated.Height;

        public void Dispose()
        {
            rotated.Dispose();
            ex.Dispose();
            ey.Dispose();
        }

        private static int Get((int L, int T, int R, int B) box, int side) =>
            side switch
            {
                0 => box.L,
                1 => box.T,
                2 => box.R,
                _ => box.B
            };

        private static (int L, int T, int R, int B) With((int L, int T, int R, int B) box, int side, int value) =>
            side switch
            {
                0 => box with { L = value },
                1 => box with { T = value },
                2 => box with { R = value },
                _ => box with { B = value }
            };

        /// <summary>
        /// Each side goes to the outermost long straight line at or beyond the content (lines
        /// must span 30% of the opposite extent); a side without one stays at the content edge.
        /// Iterated because the required length depends on the box.
        /// </summary>
        public void FindSides()
        {
            var cols   = LongestRuns(ex, true, W, H);
            var rows   = LongestRuns(ey, false, W, H);
            var margin = Math.Max(4, (int)(0.01 * Math.Max(W, H)));

            // The scan's own border is not an edge.
            for(var i = 0; i < margin; i++)
            {
                cols[i] = cols[W - 1 - i] = 0;
                rows[i] = rows[H - 1 - i] = 0;
            }

            using var exWide     = new Mat();
            using var eyWide     = new Mat();
            using var horizontal = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 1));
            using var vertical   = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(1, 3));
            Cv2.Dilate(ex, exWide, horizontal);
            Cv2.Dilate(ey, eyWide, vertical);

            const int tol = 3;
            var       box = Content;

            for(var iteration = 0; iteration < 4; iteration++)
            {
                var needX = 0.3 * (box.B - box.T);
                var needY = 0.3 * (box.R - box.L);
                var next  = Content;

                int? Outermost(int[] runs, double need, int from, int to, int step)
                {
                    for(var i = from; step > 0 ? i <= to : i >= to; i += step)
                        if(runs[i] >= need)
                            return i;

                    return null;
                }

                var candidates = new[]
                {
                    Outermost(cols, needX, 0, Math.Min(W - 1, Content.L + tol), 1),
                    Outermost(rows, needY, 0, Math.Min(H - 1, Content.T + tol), 1),
                    Outermost(cols, needX, W - 1, Math.Max(0, Content.R - tol), -1),
                    Outermost(rows, needY, H - 1, Math.Max(0, Content.B - tol), -1)
                };

                for(var side = 0; side < 4; side++)
                {
                    Lines[side] = candidates[side] is not null;

                    if(candidates[side] is not {} pos)
                    {
                        Certain[side] = false;

                        continue;
                    }

                    next          = With(next, side, pos);
                    Certain[side] = IsCertain(side, pos, box, exWide, eyWide);
                }

                if(next == box) break;

                box = next;
            }

            Box = box;
        }

        /// <summary>
        /// Whether a side line shows the physical edge rather than content: clearly outside the
        /// content, or continuous along the whole side. A line right at the content edge may be
        /// e.g. a justified text margin.
        /// </summary>
        private bool IsCertain(int side, int pos, (int L, int T, int R, int B) box, Mat exWide, Mat eyWide)
        {
            if(Math.Abs(pos - Get(Content, side)) > Math.Max(6, (int)(0.01 * Math.Max(W, H)))) return true;

            using var line = side is 0 or 2
                                 ? exWide[new Rect(pos, box.T, 1, box.B - box.T + 1)]
                                 : eyWide[new Rect(box.L, pos, box.R - box.L + 1, 1)];

            return Cv2.CountNonZero(line) >= 0.85 * line.Total();
        }

        /// <summary>
        /// On a light lid an item casts a shadow or penumbra: a grey, gradual, monotonic darkening
        /// towards the item. A side that landed in one moves in to where the profile has nearly
        /// reached the item.
        /// </summary>
        public void TrimShadows()
        {
            using var gray = new Mat();
            using var hsv  = new Mat();
            Cv2.CvtColor(rotated, gray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(rotated, hsv, ColorConversionCodes.BGR2HSV);

            using var sat = new Mat();
            Cv2.ExtractChannel(hsv, sat, 1);

            gray.GetArray(out byte[] lum);
            sat.GetArray(out byte[] chroma);

            var box = Box;

            for(var side = 0; side < 4; side++)
                if(ShadowEdge(lum, chroma, box, side) is {} edge)
                    box = With(box, side, edge);

            Box = box;
        }

        private int? ShadowEdge(byte[] lum, byte[] chroma, (int L, int T, int R, int B) box, int side)
        {
            var vertical = side is 0 or 2;
            var n        = vertical ? W : H;
            var depth    = (int)(0.15 * (vertical ? box.R - box.L : box.B - box.T));
            var edge     = Get(box, side);
            var inward   = side is 0 or 1 ? 1 : -1;

            // Profile across the side: medians over the middle 60% of its length.
            int bandFrom, bandTo;

            if(vertical)
                (bandFrom, bandTo) = (box.T + (box.B - box.T) / 5, box.B - (box.B - box.T) / 5);
            else
                (bandFrom, bandTo) = (box.L + (box.R - box.L) / 5, box.R - (box.R - box.L) / 5);

            if(bandTo <= bandFrom) return null;

            var idx = new List<int>();

            for(var k = -3; k < depth; k++)
            {
                var p = edge + inward * k;

                if(p >= 0 && p < n) idx.Add(p);
            }

            if(idx.Count < 8) return null;

            var band = new byte[bandTo - bandFrom];

            double Median(byte[] plane, int p)
            {
                for(var j = bandFrom; j < bandTo; j++) band[j - bandFrom] = vertical ? plane[j * W + p] : plane[p * W + j];

                Array.Sort(band);

                return band[band.Length / 2];
            }

            var profile = new List<double>();

            // Only the initial non-increasing stretch can be a shadow ramp.
            foreach(var p in idx)
            {
                var v = Median(lum, p);

                if(profile.Count > 0 && v - profile[^1] > 6) break;

                profile.Add(v);
            }

            if(profile.Count < 5) return null;

            var low   = profile.Min();
            var total = profile[0] - low;

            if(total < 40) return null;

            // Shadows are grey; a color fading into the paper white is artwork.
            var sats = idx.Take(profile.Count).Select(p => Median(chroma, p)).Order().ToArray();

            if(sats[sats.Length / 2] > 40) return null;

            // A shadow is gradual; a printed or physical edge goes from light to dark in 1–2 px.
            var midway = profile.Count(v => v > low + 0.25 * total && v < profile[0] - 0.25 * total);

            if(midway < 3) return null;

            var i = 0;

            while(i + 1 < profile.Count - 1 && profile[i + 1] > low + 0.25 * total) i++;

            return idx[i + 1];
        }

        /// <summary>
        /// Places sides without certain evidence at a standard inlay size measured from a side with
        /// a line, as long as all content still fits; picks the format needing the least invented
        /// margin. Keeps the box when no format fits.
        /// </summary>
        public void SnapToFormat(double dpi)
        {
            var    tol      = 3 / 25.4 * dpi;
            (int L, int T, int R, int B)? best = null;
            double bestCost = double.MaxValue;

            foreach(var (fw, fh) in FormatsMm.Concat(FormatsMm.Select(f => (f.H, f.W))))
            {
                var  output = Box;
                var  cost   = 0.0;
                var  ok     = true;

                foreach(var (axis, dimMm) in new[] { (0, fw), (1, fh) })
                {
                    int lo = axis, hi = axis + 2;
                    var dim     = dimMm / 25.4 * dpi;
                    var size    = axis == 0 ? W : H;
                    var boxLo   = Get(Box, lo);
                    var boxHi   = Get(Box, hi);
                    var contLo  = Get(Content, lo);
                    var contHi  = Get(Content, hi);
                    var options = new List<(double Cost, int Lo, int Hi)>();

                    if(Lines[lo] && Lines[hi] && Math.Abs(boxHi - boxLo - dim) <= tol) options.Add((0, boxLo, boxHi));

                    if(Lines[lo] && !Certain[hi])
                    {
                        var e = boxLo + dim;

                        if(e >= contHi - tol && e <= size - 1 + tol)
                            options.Add((e - contHi + (Certain[lo] ? 0 : tol), boxLo,
                                         (int)Math.Min(size - 1, Math.Round(e))));
                    }

                    if(Lines[hi] && !Certain[lo])
                    {
                        var e = boxHi - dim;

                        if(e >= -tol && e <= contLo + tol)
                            options.Add((contLo - e + (Certain[hi] ? 0 : tol), (int)Math.Max(0, Math.Round(e)),
                                         boxHi));
                    }

                    // No line on this axis: keep it, if the content fits the format at all.
                    if(options.Count == 0 && !Certain[lo] && !Certain[hi] && contHi - contLo <= dim + tol)
                        options.Add((tol, boxLo, boxHi));

                    if(options.Count == 0)
                    {
                        ok = false;

                        break;
                    }

                    var pick = options.MinBy(o => o.Cost);
                    output =  With(With(output, lo, pick.Lo), hi, pick.Hi);
                    cost   += pick.Cost;
                }

                if(ok && cost < bestCost)
                {
                    best     = output;
                    bestCost = cost;
                }
            }

            if(best is {} b) Box = b;
        }

        /// <summary>
        /// Remaining tilt (degrees) of the detected sides: line fit through the outermost edge
        /// pixel of every row (column) in a band around each side, length-weighted.
        /// </summary>
        public double SideResidual()
        {
            var band = Math.Max(4, (int)(0.03 * Math.Max(W, H)));

            ex.GetArray(out byte[] exPx);
            ey.GetArray(out byte[] eyPx);

            double sum = 0, weight = 0;

            for(var side = 0; side < 4; side++)
            {
                var points = new List<Point2f>();
                var at     = Get(Box, side);

                if(side is 0 or 2)
                {
                    int x0 = Math.Max(0, at - band), x1 = Math.Min(W, at + band + 1);

                    for(var y = Box.T + (Box.B - Box.T) / 10; y < Box.B - (Box.B - Box.T) / 10; y++)
                    {
                        int? hit = null;

                        for(var x = x0; x < x1; x++)
                        {
                            if(exPx[y * W + x] == 0) continue;

                            hit = x;

                            if(side == 0) break;
                        }

                        if(hit is {} hx) points.Add(new Point2f(hx, y));
                    }
                }
                else
                {
                    int y0 = Math.Max(0, at - band), y1 = Math.Min(H, at + band + 1);

                    for(var x = Box.L + (Box.R - Box.L) / 10; x < Box.R - (Box.R - Box.L) / 10; x++)
                    {
                        int? hit = null;

                        for(var y = y0; y < y1; y++)
                        {
                            if(eyPx[y * W + x] == 0) continue;

                            hit = y;

                            if(side == 1) break;
                        }

                        if(hit is {} hy) points.Add(new Point2f(x, hy));
                    }
                }

                if(points.Count < 20) continue;

                var line  = Cv2.FitLine(points, DistanceTypes.Huber, 0, 0.01, 0.01);
                var angle = Math.Atan2(line.Vy, line.Vx) * 180 / Math.PI;
                angle = ((angle + 45) % 90 + 90) % 90 - 45;

                sum    += angle * points.Count;
                weight += points.Count;
            }

            return weight > 0 ? sum / weight : 0;
        }

        public void Dump(string dir)
        {
            using var vis = rotated.Clone();

            Cv2.Rectangle(vis, new Point(Content.L, Content.T), new Point(Content.R, Content.B), Scalar.Magenta);

            Cv2.Rectangle(vis, new Point(Box.L, Box.T), new Point(Box.R, Box.B), Scalar.Lime,
                          Math.Max(1, vis.Rows / 400));

            Cv2.ImWrite(Path.Combine(dir, "inlay-detect.png"), vis);

            Console.Error.WriteLine($"    skew {Skew:0.00}° box {Box} lines " +
                                    string.Concat(Lines.Select((l, i) => l ? (Certain[i] ? "LTRB"[i] : char.ToLower("LTRB"[i])) : '-')));
        }
    }
}
