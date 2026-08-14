using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Polar unwrap of the label annulus. Used as a debug artifact and as a secondary
/// signal: text arcs laid along the circumference become horizontal runs whose
/// angular extent shows where label content concentrates.
/// </summary>
public static class PolarUnwrapper
{
    /// <summary>Unwraps the annulus: x axis = angle (0..360°), y axis = radius.</summary>
    public static Mat Unwrap(Mat gray, Disc disc, int angularResolution = 1440)
    {
        var dst        = new Mat();
        var radialSize = Math.Max(64, (int)disc.Radius);

        Cv2.WarpPolar(gray,
                      dst,
                      new Size(angularResolution, radialSize),
                      disc.Center,
                      disc.Radius,
                      InterpolationFlags.Linear,
                      WarpPolarMode.Linear);

        // WarpPolar outputs angle along rows; transpose so angle runs along columns.
        var transposed = dst.T().ToMat();
        dst.Dispose();

        return transposed;
    }
}