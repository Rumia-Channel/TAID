# TAID (ICLR 2025)

📚 [Paper](https://arxiv.org/abs/2501.16937) | 🤗 [Hugging Face](https://huggingface.co/SakanaAI) | 📝 Blog \[[EN](https://sakana.ai/taid/) | [JP](https://sakana.ai/taid-jp/)\]

<div align="center">
<img alt="overview" src="./overview.png" title="overview">
</div>

This is an official Pytorch implementation of "TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models".

## Installation

This repository now uses `uv` for environment and dependency management.

```bash
# CPU only
uv sync --extra cpu

# CUDA 12.8
uv sync --extra cuda

# ROCm 6.4 (Linux only)
uv sync --extra rocm

# Intel XPU (Linux / Windows)
uv sync --extra xpu

# Optional: DeepSpeed for multi-GPU CUDA training
uv sync --extra cuda --extra deepspeed

# make sure to login huggingface and wandb
uv run huggingface-cli login
uv run wandb login
```

`uv pip` users can also rely on automatic PyTorch backend detection because `torch-backend = "auto"` is configured for the pip-compatible interface.

FlashAttention is now optional. When it is installed and CUDA is available, the training code uses `flash_attention_2`; otherwise it falls back to `sdpa` automatically.

We conducted our original experiments in the following environment: Python 3.10.12 and CUDA 12.3 on 8 x H100 80GB. The `uv.lock` file pins a tested set of current dependencies, while the PyTorch wheel source still follows the selected backend (`cpu`, `cuda`, `rocm`, or `xpu`).

## Data Preparation

This is the script to prepare data for [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

```bash
uv run python prepare_ultrachat.py --model_type phi-3 --output_dir data
```

Qwen3.5 is also supported:

```bash
uv run python prepare_ultrachat.py --model_type qwen3.5 --output_dir data
```

`prepare_ultrachat.py` now expands each conversation into assistant-turn training samples and keeps long examples by selecting the largest fitting history window. For longer contexts, you can raise `--max_length` and `--max_output_length`; on Windows, `--num_proc` is capped automatically to avoid multiprocessing handle limits.

For datasets other than UltraChat, use `prepare_chat_dataset.py`. It accepts Hugging Face datasets or local `json/jsonl/parquet`, supports `messages`, `chosen`, ShareGPT, Alpaca, and prompt-response layouts, and lets you control `--multimodal_mode` plus `--enable_thinking`.

```bash
uv run python prepare_chat_dataset.py \
  --dataset json \
  --train_files data/custom/train.jsonl \
  --eval_files data/custom/valid.jsonl \
  --input_format messages \
  --messages_column messages \
  --model_type qwen3.5 \
  --use_processor \
  --output_name custom-qwen3.5 \
  --multimodal_mode preserve \
  --enable_thinking false
```

With `--use_processor` and `multimodal_mode=preserve`, native multimodal processors such as Qwen3.5 can precompute `pixel_values` and related vision tensors for training. `--sampling_type` is still text-only and should be left unset for multimodal batches.

## Training

We provide bash scripts for various methods in the [scripts](./scripts) directory. For example, the scripts for the experiments distilling from Llama-2 to TinyLlama can be found in [scripts/llama-2](./scripts/llama-2) directory. For instance, running the following command will execute training with TAID.

```bash
uv run bash scripts/llama-2/taid.sh
```

For direct invocation without shell scripts:

```bash
uv run python train.py \
  --teacher_model microsoft/Phi-3-mini-4k-instruct \
  --student_model TinyLlama/TinyLlama_v1.1 \
  --data_path data/phi-3 \
  --output_dir logs/phi-3-taid \
  --loss_type taid
```

Qwen3.5 example:

```bash
uv run python train.py \
  --teacher_model Qwen/Qwen3.5-9B \
  --student_model Qwen/Qwen3.5-2B \
  --data_path data/qwen3.5 \
  --output_dir logs/qwen3.5-taid \
  --loss_type taid
```

Native multimodal Qwen3.5 example:

```bash
uv run python train.py \
  --teacher_model Qwen/Qwen3.5-2B \
  --student_model Qwen/Qwen3.5-2B \
  --processor_model Qwen/Qwen3.5-2B \
  --use_processor \
  --data_path data/custom-qwen3.5 \
  --output_dir logs/qwen3.5-mm-taid \
  --loss_type taid
```

Relevant portability flags:

- `--accelerator auto|cuda|xpu|cpu`
- `--devices auto|1|0,1,2,3`
- `--strategy auto|ddp|deepspeed_stage_2`
- `--attn_implementation auto|sdpa|flash_attention_2`
- `--tokenizer_model <hf-repo-or-local-path>` when the tokenizer should differ from the teacher
- `--processor_model <hf-repo-or-local-path>` and `--use_processor` for native multimodal models
- `--trust_remote_code` for model families that still ship custom HF integrations

## Acknowledgement

We would like to thank the developers of the source models for their contributions and for making their work available.
The implmentation of baseline losses in [distil_losses](./src/distil_losses) and the sampling generation in [sampler.py](./src/sampler.py) is based on the following repositories, and we are grateful for their work.

- [DistiLLM: Towards Streamlined Distillation for Large Language Models (ICML 2024)](https://github.com/jongwooko/distillm)
- [MiniLLM: Knowledge Distillation of Large Language Models (ICLR 2024)](https://github.com/microsoft/LMOps/tree/main/minillm)

## Citation

To cite our work, you can use the following:

```bibtex
@misc{sakana2025taid,
      title         = {TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models}, 
      author.       = {Makoto Shing and Kou Misaki and Han Bao and Sho Yokoi and Takuya Akiba},
      year          = {2025},
      eprint        = {2501.16937},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG},
      url           = {https://arxiv.org/abs/2501.16937}
}
```
