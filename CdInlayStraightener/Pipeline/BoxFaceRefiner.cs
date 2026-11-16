using OpenCvSharp;

namespace CdInlayStraightener.Pipeline;

/// <summary>
/// A box face found inside the outline: <paramref name="Box"/> in the deskewed scan, and per side
/// (L, T, R, B) how far it moved inward from the outline, in millimetres.
/// </summary>
public sealed record BoxFace(Rect Box, double[] InsetMm);

/// <summary>
/// A retail box is 3D. Only its face lies on the glass; around it, the scan shows the side
/// walls rising off the glass, out of focus, as a smear of stripes along each side (and soft
/// ramps where the light falls off), then the box's silhouette edge. The outline detector
/// takes the outermost long line, which is the smear's, so the face is found by walking each
/// side inward from that outline through the smear.
///
/// Along a side the smear barely changes, so it lives in the side's luminance profile (the
/// median of each line parallel to the side): background plateaus, stripe edges and smooth
/// ramps, without the 2D texture of print. The walk follows that structure — lines no further
/// apart than a gap, and ramps on untextured ground — and stops at the last line before the
/// face: the silhouette, or the first edge that carries print texture.
/// </summary>
public static class BoxFaceRefiner
{
    private const double MaxDepthMm   = 9;   // deepest smear observed is ~6.6 mm
    private const double MaxGapMm     = 0.8; // between successive smear lines
    private const double LineStep     = 18;  // luminance step across ±StepHalfMm that is a line
    private const double StepHalfMm   = 0.085;
    private const double RampHalfMm   = 0.4;
    private const double RampSlope    = 8;   // luminance per mm that is a ramp
    private const double PlateauTol   = 6;   // luminance off the background that is still background
    private const double MaxSmearGrit = 10;  // mean along-side high-pass above which the strip is printed

    /// <summary>
    /// Scanner background luminance for box scans: the lid stays open, so it is the light
    /// ambient — the brightest part of the frame, which the box often covers half of.
    /// A dark frame (a dark cloth over the box) keeps its median.
    /// </summary>
    public static double BackgroundLuminance(Mat src)
    {
        using var gray = new Mat();
        Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

        var t      = Math.Max(4, Math.Min(gray.Rows, gray.Cols) / 100);
        var values = new List<byte>();

        foreach(var strip in new[]
                {
                    new Rect(0, 0, gray.Cols, t), new Rect(0, gray.Rows - t, gray.Cols, t), new Rect(0, 0, t, gray.Rows),
                    new Rect(gray.Cols - t, 0, t, gray.Rows)
                })
        {
            using var roi = gray[strip];

            for(var y = 0; y < roi.Rows; y += 2)
                for(var x = 0; x < roi.Cols; x += 3)
                    values.Add(roi.At<byte>(y, x));
        }

        values.Sort();

        var median = values[values.Count / 2];

        return median < 80 ? median : values[(int)(0.98 * (values.Count - 1))];
    }

    /// <summary>Moves each side of <paramref name="outline"/> inward past the box's side-wall smear.</summary>
    public static BoxFace Refine(Mat deskewed, Rect outline, double? dpi, double background)
    {
        var pxPerMm = (dpi ?? 600) / 25.4;

        using var gray = new Mat();
        Cv2.CvtColor(deskewed[outline], gray, ColorConversionCodes.BGR2GRAY);

        var insets = new int[4];

        for(var side = 0; side < 4; side++)
        {
            using var strip = SideStrip(gray, side, pxPerMm);

            insets[side] = strip is null ? 0 : Inset(strip, pxPerMm, background);
        }

        var box = new Rect(outline.X + insets[0], outline.Y + insets[1], outline.Width - insets[0] - insets[2],
                           outline.Height - insets[1] - insets[3]);

        // A box face is never this thin; something went wrong, keep the outline.
        if(box.Width < outline.Width / 2 || box.Height < outline.Height / 2) return new BoxFace(outline, new double[4]);

        return new BoxFace(box, insets.Select(i => i / pxPerMm).ToArray());
    }

    /// <summary>
    /// The band along one side (L, T, R, B) as float luminance, oriented so columns run from the
    /// outline inward and rows along the side; only the middle 60% of the side, away from corners.
    /// </summary>
    private static Mat? SideStrip(Mat gray, int side, double pxPerMm)
    {
        var depth = (int)Math.Ceiling((MaxDepthMm + 1.5) * pxPerMm);
        var along = side is 0 or 2 ? gray.Rows : gray.Cols;
        var cross = side is 0 or 2 ? gray.Cols : gray.Rows;

        depth = Math.Min(depth, cross / 3);

        var from = along / 5;
        var to   = along - along / 5;

        if(depth < 16 || to - from < 32) return null;

        var rect = side switch
        {
            0 => new Rect(0, from, depth, to - from),
            2 => new Rect(cross - depth, from, depth, to - from),
            1 => new Rect(from, 0, to - from, depth),
            _ => new Rect(from, cross - depth, to - from, depth)
        };

        var strip = new Mat();
        gray[rect].ConvertTo(strip, MatType.CV_32F);

        switch(side)
        {
            case 1:
                Cv2.Transpose(strip, strip);

                break;
            case 2:
                Cv2.Flip(strip, strip, FlipMode.Y);

                break;
            case 3:
                Cv2.Transpose(strip, strip);
                Cv2.Flip(strip, strip, FlipMode.Y);

                break;
        }

        return strip;
    }

    /// <summary>Pixels from the outline (column 0) to where the face starts.</summary>
    internal static int Inset(Mat strip, double pxPerMm, double background)
    {
        var n = strip.Cols;

        var lum  = Smooth(ColumnMedians(strip), 3);
        var grit = Smooth(AlongHighPass(strip, pxPerMm), Math.Max(3, (int)Math.Round(0.3 * pxPerMm)) | 1);

        var w     = Math.Max(1, (int)Math.Round(StepHalfMm * pxPerMm));
        var h     = Math.Max(2, (int)(RampHalfMm * pxPerMm));
        var depth = Math.Min(n - Math.Max(w, h) - 1, (int)(MaxDepthMm * pxPerMm));
        var gap   = (int)(MaxGapMm * pxPerMm);

        double Step(int i)  => i >= w && i + w < n ? Math.Abs(lum[i + w] - lum[i - w]) : 0;
        double Slope(int i) => i >= h && i + h < n ? Math.Abs(lum[i + h] - lum[i - h]) / (2.0 * h / pxPerMm) : 0;

        // The background plateau outside the box, if the outline left any.
        var k = 0;

        while(k < depth && Math.Abs(lum[k] - background) <= PlateauTol) k++;

        var  frontier = k;
        int? last     = null;

        for(var i = k; i < depth; i++)
        {
            var untextured = grit[i] <= MaxSmearGrit;
            var line       = Step(i) >= LineStep;
            var ramp       = untextured && Slope(i) >= RampSlope;

            if(line || ramp)
            {
                if(i - frontier > gap) break;

                frontier = i;

                if(line) last = i;

                // An edge with print texture is the face's own: stop right after it.
                if(line && !untextured) break;
            }
            else if(!untextured && i > frontier)
                break;
        }

        if(last is not {} edge) return Math.Min(k, depth);

        while(edge < n - 1 && Step(edge) >= LineStep) edge++;

        return Math.Min(depth, edge);
    }

    private static double[] ColumnMedians(Mat strip)
    {
        strip.GetArray(out float[] px);

        int rows = strip.Rows, cols = strip.Cols;
        var col    = new float[rows];
        var result = new double[cols];

        for(var x = 0; x < cols; x++)
        {
            for(var y = 0; y < rows; y++) col[y] = px[y * cols + x];

            Array.Sort(col);
            result[x] = rows % 2 == 1 ? col[rows / 2] : (col[rows / 2 - 1] + col[rows / 2]) / 2.0;
        }

        return result;
    }

    /// <summary>Per column, the mean absolute high-pass along the side: print texture, absent in the smear.</summary>
    private static double[] AlongHighPass(Mat strip, double pxPerMm)
    {
        var sigma = 8 * pxPerMm / (600 / 25.4);
        var size  = 2 * (int)Math.Ceiling(4 * sigma) + 1;

        using var kx      = new Mat(1, 1, MatType.CV_32F, Scalar.All(1));
        using var ky      = Cv2.GetGaussianKernel(size, sigma, MatType.CV_32F)!;
        using var smooth  = new Mat();
        using var hp      = new Mat();
        using var reduced = new Mat();

        Cv2.SepFilter2D(strip, smooth, MatType.CV_32F, kx, ky, borderType: BorderTypes.Reflect);
        Cv2.Absdiff(strip, smooth, hp);
        Cv2.Reduce(hp, reduced, ReduceDimension.Row, ReduceTypes.Avg, MatType.CV_32F);

        reduced.GetArray(out float[] values);

        return values.Select(v => (double)v).ToArray();
    }

    /// <summary>Centered moving average with an odd window, shrinking at the ends.</summary>
    private static double[] Smooth(double[] v, int window)
    {
        var r      = window / 2;
        var result = new double[v.Length];

        for(var i = 0; i < v.Length; i++)
        {
            int lo = Math.Max(0, i - r), hi = Math.Min(v.Length - 1, i + r);
            var s  = 0.0;

            for(var j = lo; j <= hi; j++) s += v[j];

            result[i] = s / (hi - lo + 1);
        }

        return result;
    }
}
