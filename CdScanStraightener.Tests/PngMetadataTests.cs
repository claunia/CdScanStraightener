using System.Buffers.Binary;
using CdScanStraightener.Io;
using OpenCvSharp;

namespace CdScanStraightener.Tests;

public sealed class PngMetadataTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("cdscan-tests").FullName;

    public void Dispose() => Directory.Delete(_dir, true);

    private string WritePng(string name)
    {
        var       path = Path.Combine(_dir, name);
        using var m    = new Mat(8, 8, MatType.CV_8UC3, Scalar.Blue);
        Cv2.ImWrite(path, m);

        return path;
    }

    private static byte[] MakeChunk(string type, byte[] data)
    {
        var chunk = new byte[12 + data.Length];
        BinaryPrimitives.WriteUInt32BigEndian(chunk, (uint)data.Length);
        System.Text.Encoding.ASCII.GetBytes(type).CopyTo(chunk, 4);
        data.CopyTo(chunk, 8);
        var crc = System.IO.Hashing.Crc32.Hash(chunk[4..^4]);

        // Crc32.Hash returns little-endian; PNG wants big-endian.
        chunk[^4] = crc[3];
        chunk[^3] = crc[2];
        chunk[^2] = crc[1];
        chunk[^1] = crc[0];

        return chunk;
    }

    [Fact]
    public void PhysChunkRoundtrips()
    {
        var src = WritePng("src.png");
        var dst = WritePng("dst.png");

        // pHYs: 11811 px/m both axes (300 DPI), unit = meter.
        var data = new byte[9];
        BinaryPrimitives.WriteUInt32BigEndian(data,           11811);
        BinaryPrimitives.WriteUInt32BigEndian(data.AsSpan(4), 11811);
        data[8] = 1;
        var phys = MakeChunk("pHYs", data);

        PngMetadata.WritePreservedChunks(src, [phys]);
        var chunks = PngMetadata.ReadPreservedChunks(src);
        Assert.Single(chunks);
        Assert.Equal(phys, chunks[0]);

        PngMetadata.WritePreservedChunks(dst, chunks);
        Assert.Equal(phys, Assert.Single(PngMetadata.ReadPreservedChunks(dst)));

        // File must still be a decodable PNG.
        using var reread = Cv2.ImRead(dst);
        Assert.False(reread.Empty());
        Assert.Equal(8, reread.Rows);
    }

    [Fact]
    public void NoMetadataMeansNoChunks()
    {
        var src = WritePng("plain.png");
        Assert.Empty(PngMetadata.ReadPreservedChunks(src));
    }

    [Fact]
    public void NonPngThrows()
    {
        var path = Path.Combine(_dir, "not.png");
        File.WriteAllBytes(path, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        Assert.Throws<InvalidDataException>(() => PngMetadata.ReadPreservedChunks(path));
    }
}