# Transformer PyTorch Implementation

PyTorch implementation of the Transformer model described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin (2017/2023).

## Required dependencies
- Python (ver. 3.12 was used)
- Poetry (ver. 1.8.5 was used)

## Install dependencies
### Install with CUDA (when using GPU)
```bash
poetry install --with torch-cuda
```

### Install without CUDA (CPU only)
```bash
poetry install --with torch-cpu
```
