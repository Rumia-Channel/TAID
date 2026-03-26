# TAID (ICLR 2025)

📚 [論文](https://arxiv.org/abs/2501.16937) | 🤗 [Hugging Face](https://huggingface.co/SakanaAI) | 📝 Blog [[EN](https://sakana.ai/taid/) | [JP](https://sakana.ai/taid-jp/)]

<div align="center">
<img alt="overview" src="./overview.png" title="overview">
</div>

このリポジトリは、"TAID: Temporally Adaptive Interpolated Distillation for Efficient Knowledge Transfer in Language Models" の公式 PyTorch 実装です。

## インストール

このリポジトリでは、依存管理と実行環境管理に `uv` を使用します。

```bash
# CPU のみ
uv sync --extra cpu

# CUDA 12.8
uv sync --extra cuda

# ROCm 6.4 (Linux のみ)
uv sync --extra rocm

# Intel XPU (Linux / Windows)
uv sync --extra xpu

# 任意: 複数 GPU の CUDA 学習で DeepSpeed を使う場合
uv sync --extra cuda --extra deepspeed

# Hugging Face と Weights & Biases にログイン
uv run huggingface-cli login
uv run wandb login
```

`uv pip` を使う場合でも、`torch-backend = "auto"` を設定してあるため、PyTorch のバックエンドは自動判定されます。

FlashAttention は必須ではありません。CUDA が利用可能で `flash_attn` が入っている場合のみ `flash_attention_2` を使い、それ以外では自動で `sdpa` にフォールバックします。

元論文の実験環境は Python 3.10.12 / CUDA 12.3 / H100 80GB x8 です。現在は `uv.lock` に検証済みの依存セットを固定しつつ、PyTorch wheel は `cpu` / `cuda` / `rocm` / `xpu` の選択に応じて切り替わります。

## データ準備

[Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 向けのデータ準備例:

```bash
uv run python prepare_ultrachat.py --model_type phi-3 --output_dir data
```

Qwen3.5 もサポートしています:

```bash
uv run python prepare_ultrachat.py --model_type qwen3.5 --output_dir data
```

`prepare_ultrachat.py` は会話を assistant 応答単位の学習サンプルへ展開し、長い会話でも入る最大の履歴窓を選ぶようになっています。より長い文脈を残したい場合は `--max_length` と `--max_output_length` を調整してください。Windows では multiprocessing の上限回避のため `--num_proc` が自動で抑えられます。

UltraChat 以外のデータセットには `prepare_chat_dataset.py` を使ってください。Hugging Face dataset と local の `json/jsonl/parquet` に対応し、`messages`、`chosen`、ShareGPT、Alpaca、prompt-response を正規化できます。`--multimodal_mode` と `--enable_thinking` も指定できます。

```bash
uv run python prepare_chat_dataset.py \
  --dataset json \
  --train_files data/custom/train.jsonl \
  --eval_files data/custom/valid.jsonl \
  --input_format messages \
  --messages_column messages \
  --model_type qwen3.5 \
  --output_name custom-qwen3.5 \
  --multimodal_mode preserve \
  --enable_thinking false
```

`multimodal_mode=preserve` は chat template が出す modality token を保持します。現状の学習器自体は text-only なので、画像 tensor そのものではなく特殊 token を残す挙動です。

## 学習

各手法の実行スクリプトは [scripts](./scripts) 配下にあります。たとえば Llama-2 系の TAID 実験は [scripts/llama-2](./scripts/llama-2) にあり、次のコマンドで実行できます。

```bash
uv run bash scripts/llama-2/taid.sh
```

シェルスクリプトを使わず直接実行する場合:

```bash
uv run python train.py \
  --teacher_model microsoft/Phi-3-mini-4k-instruct \
  --student_model TinyLlama/TinyLlama_v1.1 \
  --data_path data/phi-3 \
  --output_dir logs/phi-3-taid \
  --loss_type taid
```

Qwen3.5 の例:

```bash
uv run python train.py \
  --teacher_model Qwen/Qwen3.5-9B \
  --student_model Qwen/Qwen3.5-2B \
  --data_path data/qwen3.5 \
  --output_dir logs/qwen3.5-taid \
  --loss_type taid
```

主な可搬性関連フラグ:

- `--accelerator auto|cuda|xpu|cpu`
- `--devices auto|1|0,1,2,3`
- `--strategy auto|ddp|deepspeed_stage_2`
- `--attn_implementation auto|sdpa|flash_attention_2`
- `--tokenizer_model <hf-repo-or-local-path>`: tokenizer を teacher と分けたい場合
- `--trust_remote_code`: 独自 HF 実装を使うモデル系を読み込む場合

## 謝辞

公開済みのソースモデルを提供している開発者の皆さまに感謝します。
[distil_losses](./src/distil_losses) のベースライン損失実装と [sampler.py](./src/sampler.py) のサンプリング生成は、以下の実装を参考にしています。

- [DistiLLM: Towards Streamlined Distillation for Large Language Models (ICML 2024)](https://github.com/jongwooko/distillm)
- [MiniLLM: Knowledge Distillation of Large Language Models (ICLR 2024)](https://github.com/microsoft/LMOps/tree/main/minillm)

## 引用

引用には以下を使用してください。

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
