# embzip 🗜️

A Python package for efficiently compressing and decompressing embeddings using Product Quantization.

## Installation

```bash
pip install embzip
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Quantize embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Quantize the embeddings with default parameters
quantized = embzip.quantize(embeddings)

# Quantize with custom M parameter (controls compression level)
# Lower M = higher compression but lower accuracy
quantized_high_compression = embzip.quantize(embeddings, m=24)  # more compressed
quantized_high_quality = embzip.quantize(embeddings, m=96)      # less compressed
```

### Save and load compressed embeddings

```python
import torch
import embzip

# Create sample embeddings
embeddings = torch.randn(1000, 768)  # 1000 vectors with 768 dimensions

# Note: You probably don't want to use random vectors, because they 
# won't compress well (e.g. you'll either get little savings in space 
# or bad cosine similarity). You should use real embeddings, such as
# those from a sentence-transformers model.

# Save embeddings to a file with default parameters
embzip.save(embeddings, "embeddings.ezip")

# Save with custom M parameter
embzip.save(embeddings, "embeddings_higher_compression.ezip", m=24)

# Load embeddings from a file
loaded_embeddings = embzip.load("embeddings.ezip")

# Check similarity between original and reconstructed embeddings
similarity = torch.nn.functional.cosine_similarity(
    embeddings.view(-1), loaded_embeddings.view(-1), dim=0
)
print(f"Cosine similarity: {similarity.item()}")
```

## Parameter Tuning for PQ

The `m` parameter controls the number of sub-quantizers used in product quantization. Lower `m` means higher compression, lower performance Higher `m` indicates lower compression (higher accuracy). The default `m` is calculated as `dimension // 16`.

Product Quantization works by splitting each embedding vector into M equal sub-vectors and quantizing each sub-vector independently using a small codebook. When M is higher, more sub-quantizers are used and each sub-vector becomes smaller (fewer dimensions), resulting in a more accurate representation but larger storage requirements. When M is lower, fewer sub-quantizers cover more dimensions each, providing higher compression but lower accuracy.

For a 768-dimension vector, the default M would be 48, which tends to provide a reasonable balance between compression and accuracy for most applications.

## Benchmarks

Here are some example benchmarking numbers for 1000 text embeddings embedded using [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

```bash
====================================================================================================
Model: all-MiniLM-L6-v2 | n: 1000 | m: 64
Original size: 1536000 bytes
Compressed size: 56372 bytes
Compression ratio: 27.25
Cosine similarity: 0.7882668375968933
====================================================================================================

====================================================================================================
Model: all-MiniLM-L6-v2 | n: 1000 | m: 128
Original size: 1536000 bytes
Compressed size: 87920 bytes
Compression ratio: 17.47
Cosine similarity: 0.9056953191757202
====================================================================================================

====================================================================================================
Model: all-MiniLM-L6-v2 | n: 1000 | m: 384
Original size: 1536000 bytes
Compressed size: 216112 bytes
Compression ratio: 7.11
Cosine similarity: 0.9953747391700745
====================================================================================================
```

## Requirements

- Python 3.6+
- PyTorch
- FAISS

## Development

To run tests:
```bash
pytest
``` 



### Citation

```
@misc{morris2025embzip,
  author = {Morris, John X},
  title = {embzip: A Python package for efficiently compressing and decompressing embeddings using Product Quantization},
  url = {https://github.com/jxmorris12/embzip},
  year = {2025}
}
```
