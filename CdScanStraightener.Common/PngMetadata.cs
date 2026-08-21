using System.Buffers.Binary;

namespace CdScanStraightener.Common;

/// <summary>
/// Minimal PNG chunk handling to carry ancillary metadata — pHYs (DPI), iCCP (ICC profile),
/// sRGB, gAMA, cHRM, eXIf (EXIF/IFD0) and the textual chunks — from a source PNG to an
/// output PNG, since OpenCV's encoder drops them.
/// </summary>
public static class PngMetadata
{
    private static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>Chunk types that appear once, in the order they should be written after IHDR.</summary>
    private static readonly string[] SingleTypes = ["iCCP", "sRGB", "gAMA", "cHRM", "pHYs", "eXIf"];

    /// <summary>Textual chunk types, which may legitimately appear many times and are all kept, in source order.</summary>
    private static readonly string[] RepeatableTypes = ["tEXt", "zTXt", "iTXt"];

    private static bool IsPreserved(string type) => SingleTypes.Contains(type) || RepeatableTypes.Contains(type);

    /// <summary>Returns the raw preserved chunks (length+type+data+crc each) of a PNG, in canonical order.</summary>
    public static List<byte[]> ReadPreservedChunks(string path)
    {
        var bytes      = File.ReadAllBytes(path);
        var single     = new Dictionary<string, byte[]>();
        var repeatable = new List<byte[]>();

        foreach(var (type, offset, totalLen) in EnumerateChunks(bytes))
        {
            var chunk = bytes.AsSpan(offset, totalLen).ToArray();

            if(RepeatableTypes.Contains(type))
                repeatable.Add(chunk);
            else if(SingleTypes.Contains(type) && !single.ContainsKey(type))
                single[type] = type == "eXIf" ? NormalizeExifOrientation(chunk) : chunk;
        }

        return [.. SingleTypes.Where(single.ContainsKey).Select(t => single[t]), .. repeatable];
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
            if(IsPreserved(type)) continue; // replaced by the source's chunks
            ms.Write(bytes, offset, totalLen);

            if(type == "IHDR")
                foreach(var chunk in chunks)
                    ms.Write(chunk, 0, chunk.Length);
        }

        File.WriteAllBytes(path, ms.ToArray());
    }

    /// <summary>
    /// Returns the eXIf chunk with IFD0's Orientation tag forced to 1 (normal), CRC recomputed.
    /// Everything else — make, model, software, timestamps, the EXIF sub-IFD — is carried over
    /// untouched. The straightened pixels already are the intended orientation, so a viewer
    /// honouring an inherited non-normal Orientation would rotate the image a second time.
    /// </summary>
    private static byte[] NormalizeExifOrientation(byte[] chunk)
    {
        const int  headerLen    = 8;  // 4-byte length + 4-byte type
        const int  tiffHeader   = 8;  // byte order + magic + IFD0 offset
        const int  entryLen     = 12; // tag + type + count + value
        const ushort orientation = 0x0112;

        var dataLen = (int)BinaryPrimitives.ReadUInt32BigEndian(chunk.AsSpan(0, 4));

        if(dataLen < tiffHeader) return chunk;

        // TIFF header: "MM" big-endian or "II" little-endian. Anything else is not EXIF
        // we understand, so leave the chunk exactly as it was.
        var tiff = headerLen; // absolute offset of the TIFF data within the chunk

        if(chunk[tiff] != chunk[tiff + 1] || (chunk[tiff] != (byte)'M' && chunk[tiff] != (byte)'I')) return chunk;

        var bigEndian = chunk[tiff] == (byte)'M';

        ushort ReadU16(int at) =>
            bigEndian
                ? BinaryPrimitives.ReadUInt16BigEndian(chunk.AsSpan(tiff + at, 2))
                : BinaryPrimitives.ReadUInt16LittleEndian(chunk.AsSpan(tiff + at, 2));

        uint ReadU32(int at) =>
            bigEndian
                ? BinaryPrimitives.ReadUInt32BigEndian(chunk.AsSpan(tiff + at, 4))
                : BinaryPrimitives.ReadUInt32LittleEndian(chunk.AsSpan(tiff + at, 4));

        var ifd0 = (int)ReadU32(4);

        if(ifd0 + 2 > dataLen) return chunk;

        var count = ReadU16(ifd0);

        for(var i = 0; i < count; i++)
        {
            var entry = ifd0 + 2 + i * entryLen;

            if(entry + entryLen > dataLen) break;
            if(ReadU16(entry) != orientation) continue;

            var value = chunk.AsSpan(tiff + entry + 8, 2);

            if(bigEndian)
                BinaryPrimitives.WriteUInt16BigEndian(value, 1);
            else
                BinaryPrimitives.WriteUInt16LittleEndian(value, 1);

            // The data changed, so the chunk's CRC (over type + data) must be recomputed.
            BinaryPrimitives.WriteUInt32BigEndian(chunk.AsSpan(headerLen + dataLen, 4),
                                                  Crc32(chunk.AsSpan(4, 4 + dataLen)));

            return chunk;
        }

        return chunk;
    }

    private static readonly uint[] CrcTable = BuildCrcTable();

    private static uint[] BuildCrcTable()
    {
        var table = new uint[256];

        for(uint n = 0; n < 256; n++)
        {
            var c = n;

            for(var k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;

            table[n] = c;
        }

        return table;
    }

    private static uint Crc32(ReadOnlySpan<byte> bytes)
    {
        var c = 0xFFFFFFFFu;

        foreach(var b in bytes) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);

        return c ^ 0xFFFFFFFFu;
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
