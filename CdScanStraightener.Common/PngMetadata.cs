using System.Buffers.Binary;

namespace CdScanStraightener.Common;

/// <summary>
/// Minimal PNG chunk handling to carry ancillary metadata — pHYs (DPI), iCCP (ICC profile),
/// sRGB, gAMA, cHRM — from a source PNG to an output PNG, since OpenCV's encoder drops them.
/// </summary>
public static class PngMetadata
{
    private static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>Chunk types worth preserving, in the order they should appear after IHDR.</summary>
    private static readonly string[] PreservedTypes = ["iCCP", "sRGB", "gAMA", "cHRM", "pHYs"];

    /// <summary>Returns the raw preserved chunks (length+type+data+crc each) of a PNG, in canonical order.</summary>
    public static List<byte[]> ReadPreservedChunks(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var found = new Dictionary<string, byte[]>();

        foreach(var (type, offset, totalLen) in EnumerateChunks(bytes))
        {
            if(PreservedTypes.Contains(type) && !found.ContainsKey(type))
                found[type] = bytes.AsSpan(offset, totalLen).ToArray();
        }

        return PreservedTypes.Where(found.ContainsKey).Select(t => found[t]).ToList();
    }

    /// <summary>
    /// Inserts the given raw chunks into the PNG at <paramref name="path"/> immediately after IHDR,
    /// dropping any chunks of the same types the encoder may have written, and rewrites the file.
    /// </summary>
    public static void WritePreservedChunks(string path, IReadOnlyList<byte[]> chunks)
    {
        if(chunks.Count == 0) return;
        var       bytes = File.ReadAllBytes(path);
        using var ms    = new MemoryStream(bytes.Length + chunks.Sum(c => c.Length));
        ms.Write(bytes, 0, Signature.Length);

        foreach(var (type, offset, totalLen) in EnumerateChunks(bytes))
        {
            if(PreservedTypes.Contains(type)) continue; // replaced by the source's chunks
            ms.Write(bytes, offset, totalLen);

            if(type == "IHDR")
                foreach(var chunk in chunks)
                    ms.Write(chunk, 0, chunk.Length);
        }

        File.WriteAllBytes(path, ms.ToArray());
    }

    private static IEnumerable<(string Type, int Offset, int TotalLen)> EnumerateChunks(byte[] bytes)
    {
        if(bytes.Length < Signature.Length || !bytes.AsSpan(0, Signature.Length).SequenceEqual(Signature))
            throw new InvalidDataException("Not a PNG file.");

        var pos = Signature.Length;

        while(pos + 12 <= bytes.Length)
        {
            var dataLen = (int)BinaryPrimitives.ReadUInt32BigEndian(bytes.AsSpan(pos, 4));
            var type    = System.Text.Encoding.ASCII.GetString(bytes, pos + 4, 4);
            var total   = 12 + dataLen;

            if(pos + total > bytes.Length) throw new InvalidDataException("Truncated PNG chunk.");

            yield return (type, pos, total);

            if(type == "IEND") yield break;
            pos += total;
        }
    }
}