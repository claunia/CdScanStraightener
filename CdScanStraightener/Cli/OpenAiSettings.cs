namespace CdScanStraightener.Cli;

/// <summary>
/// Settings for the vision-model orientation fallback, bound from the "OpenAI" section of
/// appsettings.json. Works with api.openai.com or any OpenAI-compatible server (LM Studio,
/// Ollama, vLLM…) — point BaseUrl at the server's /v1 root. Local servers need no ApiKey.
/// </summary>
public sealed record OpenAiSettings
{
    public bool Enabled { get; init; }
    public string BaseUrl { get; init; } = "https://api.openai.com/v1";
    public string? ApiKey { get; init; }
    public string Model { get; init; } = "gpt-4o-mini";

    /// <summary>Model used for upright YES/NO verification; defaults to <see cref="Model"/>.</summary>
    public string? VerifyModel { get; init; }

    /// <summary>Maximum simultaneous requests to the endpoint.</summary>
    public int MaxParallelRequests { get; init; } = 4;

    public bool IsUsable => Enabled && (!string.IsNullOrEmpty(ApiKey) ||
                                        !BaseUrl.StartsWith("https://api.openai.com", StringComparison.OrdinalIgnoreCase));
}
