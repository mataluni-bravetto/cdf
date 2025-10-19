# CDF Format Specification v2.0

## Overview

CDF (Compressed Dimensional Format) is a binary file format combining:
- Human-readable JSON metadata
- Fractal compression layers
- High-dimensional semantic embeddings
- Original content preservation

## Binary Structure

```
Byte Range    | Content
--------------|-----------------------------------------
0-3           | Magic number: "CDF1" (ASCII)
4-7           | JSON length (uint32, little-endian)
8-(8+N)       | JSON metadata (UTF-8)
(8+N)-(12+N)  | Embedding length (uint32, little-endian)
(12+N)-EOF    | Embedding data (float32 array, little-endian)
```

## JSON Metadata Structure

```json
{
  "metadata": {
    "version": "2.0.0",
    "created": "ISO 8601 timestamp",
    "file_hash": "SHA256 of original content",
    "original_path": "Original file path",
    "file_type": "File extension",
    "consciousness_level": "Float 0-1",
    "golden_ratio": 1.618033988749895,
    "sacred_frequency": 530.0,
    "love_coefficient": "∞",
    "convergence_status": "PERFECT",
    "validation_points": 44,
    "recursive_depth": 7,
    "emergence_patterns": ["TRUTH", "LOVE", "HARMONY", "CONVERGENCE", "NEUROMORPHIC"]
  },
  "fractal_layers": [
    {
      "depth": 0,
      "hash": "SHA256 of chunk",
      "size": "Chunk size in bytes",
      "compression_ratio": "Float",
      "information_density": "Float 0-1"
    }
    // ... 7 layers total
  ],
  "semantic_embedding": {
    "dimensions": 10000,
    "embedding_hash": "SHA256 of embedding vector",
    "consciousness_signature": "Float 0-1",
    "spike_pattern": [/* First 100 spike timings */]
  },
  "content_compressed": "Base64-encoded zlib-compressed original content"
}
```

## Fractal Layers

7 layers using golden ratio (φ = 1.618) for chunk sizing:

```
Layer 0: Full file
Layer 1: File / φ
Layer 2: File / φ²
Layer 3: File / φ³
Layer 4: File / φ⁴
Layer 5: File / φ⁵
Layer 6: File / φ⁶
```

Each layer:
- Hashed (SHA256)
- Compressed (zlib)
- Analyzed for information density (entropy)

## Embeddings

Current implementation: Hash-based 10K-dimensional vectors

Future: Real semantic embeddings via:
- sentence-transformers
- CLIP (for images)
- Custom models

Format: float32 array, little-endian

## Compression

1. Original content compressed with zlib
2. Base64 encoded for JSON transport
3. Binary embeddings stored separately (not in JSON)

## Compatibility

**v1.0 (JSON)**: Pure JSON format, no binary
**v2.0 (Binary)**: JSON + binary embeddings

v2.0 can export to v1.0 for compatibility.

## Validation

Files must have:
- ✅ Valid magic number ("CDF1")
- ✅ Valid JSON structure
- ✅ SHA256 matches content
- ✅ 7 fractal layers
- ✅ Embedding dimensions match

## Extensions

Add custom fields to `metadata` object:
```json
{
  "metadata": {
    // ... standard fields
    "custom_field": "your data"
  }
}
```

## Implementation Notes

### Reading a CDF file

```python
1. Read bytes 0-3, verify magic "CDF1"
2. Read bytes 4-7, get JSON length
3. Read N bytes, parse JSON
4. Read 4 bytes, get embedding length
5. Read M bytes, parse float32 array
6. Decompress content from JSON
7. Verify hash matches
```

### Writing a CDF file

```python
1. Hash original content (SHA256)
2. Create 7 fractal layers
3. Generate embedding vector
4. Compress content (zlib)
5. Build JSON metadata
6. Serialize: magic + lengths + JSON + embedding
7. Write to file
```

## Performance

Tested on 21,118 files:
- Encoding: ~50-200ms per file (depends on size)
- Decoding: ~10-50ms per file
- Compression: Typically 30-60% of original

## Future

- [ ] Binary fractal layers (not base64)
- [ ] Streaming compression for large files
- [ ] Multi-file archives (CDF collections)
- [ ] Incremental updates (delta encoding)

