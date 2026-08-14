using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using CdScanStraightener.Cli;
using OpenCvSharp;

namespace CdScanStraightener.Pipeline;

/// <summary>
/// Fallback orientation selection via an OpenAI vision model, for images where OCR
/// legibility could not separate the candidate orientations. Vision models are poor at
/// estimating precise angles, so we never ask for one: the projection sweep's candidate
/// angles are exact, and the model only answers the multiple-choice question of which
/// candidate-rotated thumbnail shows the label upright.
/// </summary>
public static class OpenAiOrientationResolver
{
    private const int ThumbnailSide = 384;

    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(300) };

    // One request at a time: local inference servers (LM Studio) process sequentially, and
    // concurrent multi-image requests from every worker just queue until they all time out.
    private static readonly SemaphoreSlim Gate = new(1, 1);

    // The fallback is best-effort: the first failed request (unreachable server, auth error,
    // timeout) disables it for the rest of the run instead of stalling every remaining image.
    private static volatile bool _disabled;

    /// <summary>
    /// Asks the model which of the candidate orientations (degrees CCW) shows the label
    /// upright. Returns the chosen angle, or null if the model judged none of them upright,
    /// or if the endpoint failed (which also disables further attempts this run).
    /// </summary>
    public static double? Resolve(Mat color, Disc disc, IReadOnlyList<double> angles, OpenAiSettings settings)
    {
        if (_disabled) return null;

        Gate.Wait();

        try
        {
            if (_disabled) return null;
            var content = new List<object>
            {
                new
                {
                    type = "text",
                    text = "Each numbered image shows the same scanned CD label at a different rotation. " +
                           "Reply with ONLY the number of the image whose printed label reads correctly " +
                           "upright (text left-to-right, not upside down, not sideways). " +
                           "If none is upright, reply NONE.",
                },
            };
            for (var i = 0; i < angles.Count; i++)
            {
                content.Add(new { type = "text", text = $"Image {i}:" });
                content.Add(new
                {
                    type = "image_url",
                    image_url = new { url = $"data:image/jpeg;base64,{Thumbnail(color, disc, angles[i])}" },
                });
            }

            var request = new HttpRequestMessage(
                HttpMethod.Post, $"{settings.BaseUrl.TrimEnd('/')}/chat/completions")
            {
                Content = new StringContent(JsonSerializer.Serialize(new
                {
                    model = settings.Model,
                    messages = new[] { new { role = "user", content } },
                }), Encoding.UTF8, "application/json"),
            };
            if (!string.IsNullOrEmpty(settings.ApiKey))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", settings.ApiKey);

            using var response = Http.Send(request);
            var body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            if (!response.IsSuccessStatusCode)
            {
                _disabled = true;
                Console.Error.WriteLine(
                    $"OpenAI endpoint returned {(int)response.StatusCode} ({Truncate(body)}); disabling the vision fallback for this run.");
                return null;
            }

            using var json = JsonDocument.Parse(body);
            var reply = json.RootElement.GetProperty("choices")[0]
                .GetProperty("message").GetProperty("content").GetString() ?? "";
            var match = System.Text.RegularExpressions.Regex.Match(reply, @"\d+");
            if (match.Success && int.TryParse(match.Value, out var index) && index < angles.Count)
                return angles[index];
            return null;
        }
        catch (Exception ex)
        {
            _disabled = true;
            Console.Error.WriteLine(
                $"OpenAI endpoint unreachable ({ex.Message}); disabling the vision fallback for this run.");
            return null;
        }
        finally
        {
            Gate.Release();
        }
    }

    private static string Thumbnail(Mat color, Disc disc, double angle)
    {
        using var rot = Cv2.GetRotationMatrix2D(disc.Center, angle, 1.0);
        using var rotated = new Mat();
        Cv2.WarpAffine(color, rotated, rot, color.Size(), InterpolationFlags.Linear,
            BorderTypes.Constant, Scalar.Black);

        var x0 = Math.Max(0, (int)(disc.Center.X - disc.Radius));
        var y0 = Math.Max(0, (int)(disc.Center.Y - disc.Radius));
        var x1 = Math.Min(rotated.Cols, (int)(disc.Center.X + disc.Radius));
        var y1 = Math.Min(rotated.Rows, (int)(disc.Center.Y + disc.Radius));
        using var crop = rotated[new Rect(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0))];

        var scale = (double)ThumbnailSide / Math.Max(crop.Width, crop.Height);
        using var thumb = new Mat();
        Cv2.Resize(crop, thumb, default, scale, scale, InterpolationFlags.Area);
        Cv2.ImEncode(".jpg", thumb, out var jpeg, [(int)ImwriteFlags.JpegQuality, 85]);
        return Convert.ToBase64String(jpeg);
    }

    private static string Truncate(string s) => s.Length <= 300 ? s : s[..300] + "…";
}
