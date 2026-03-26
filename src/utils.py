import os
import re
from typing import List
from inspect import isfunction
import itertools
import glob
from natsort import natsorted

import torch
from torch import nn
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file as load_safetensors


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def flatten_list(x):
    return list(itertools.chain.from_iterable(x))


def normalize_chat_text(text: str) -> str:
    text = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


GPT_OSS_MESSAGE_RE = re.compile(
    r"<\|start\|>assistant"
    r"(?:<\|channel\|>(?P<channel>[^<]+))?"
    r"(?:<\|recipient\|>[^<]+)?"
    r"<\|message\|>(?P<body>.*?)(?=(?:<\|end\|>|<\|return\|>|$))",
    flags=re.DOTALL,
)
GPT_OSS_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")


def get_text_tokenizer(preprocessor):
    return preprocessor.tokenizer if hasattr(preprocessor, "tokenizer") else preprocessor


def get_pad_token_id(preprocessor):
    tokenizer = get_text_tokenizer(preprocessor)
    return tokenizer.pad_token_id


def get_eos_token_id(preprocessor):
    tokenizer = get_text_tokenizer(preprocessor)
    return tokenizer.eos_token_id


def is_gpt_oss_preprocessor(preprocessor) -> bool:
    tokenizer = get_text_tokenizer(preprocessor)
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "gpt-oss" in name:
        return True
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "<|start|>assistant" in chat_template and "Reasoning:" in chat_template


def extract_gpt_oss_visible_text(text: str) -> str:
    final_parts = []
    fallback_parts = []
    for match in GPT_OSS_MESSAGE_RE.finditer(text):
        channel = (match.group("channel") or "").strip()
        body = GPT_OSS_SPECIAL_TOKEN_RE.sub("", match.group("body")).strip()
        if not body:
            continue
        if channel == "final":
            final_parts.append(body)
        else:
            fallback_parts.append(body)

    selected = final_parts or fallback_parts
    if selected:
        return "\n".join(selected).strip()

    stripped = GPT_OSS_SPECIAL_TOKEN_RE.sub("", text)
    stripped = re.sub(r"assistant(?:analysis|commentary|final)+", "", stripped)
    return stripped.strip()


def decode_generated_texts(preprocessor, generated_ids: torch.Tensor) -> List[str]:
    tokenizer = get_text_tokenizer(preprocessor)
    if is_gpt_oss_preprocessor(preprocessor):
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        return [normalize_chat_text(extract_gpt_oss_visible_text(text)) for text in decoded]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [normalize_chat_text(text) for text in decoded]


def get_generated_ids(generated_ids: torch.Tensor, input_ids: torch.Tensor):
    input_len = input_ids.shape[-1]
    return generated_ids[:, input_len:]


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def get_decay_parameter_names(model) -> List[str]:
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    no_decay = ["bias"]
    decay_parameters = [
        name for name in decay_parameters if not any(nd in name for nd in no_decay)
    ]
    return decay_parameters


def get_optimizer_params(model: nn.Module, loss_fn: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    decay_parameters = get_decay_parameter_names(model)

    if loss_fn is not None:
        param_optimizer += list(loss_fn.named_parameters())
        decay_parameters += get_decay_parameter_names(loss_fn)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters


def load_tokenizer(tokenizer_path: str, **tokenizer_kwargs):
    tokenizer_name = tokenizer_path.lower()
    if "phi-3" in tokenizer_name:
        tokenizer_kwargs["pad_token"] = "<unk>"
        tokenizer_kwargs["padding_side"] = "right"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    if "qwen" in tokenizer_name and tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_preprocessor(
    preprocessor_path: str,
    use_processor: bool = False,
    **preprocessor_kwargs,
):
    if not use_processor:
        return load_tokenizer(preprocessor_path, **preprocessor_kwargs)

    processor = AutoProcessor.from_pretrained(preprocessor_path, **preprocessor_kwargs)
    tokenizer = get_text_tokenizer(processor)
    tokenizer_name = preprocessor_path.lower()
    if "phi-3" in tokenizer_name:
        tokenizer.pad_token = "<unk>"
        tokenizer.padding_side = "right"
    if "qwen" in tokenizer_name and tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def get_best_checkpoint_name(logdir):
    ckpt = os.path.join(logdir, "last**.ckpt")
    ckpt = natsorted(glob.glob(ckpt))
    if len(ckpt) == 0:
        ckpt = os.path.join(logdir, "epoch**.ckpt")
        ckpt = natsorted(glob.glob(ckpt))
    ckpt = ckpt[-1]
    return ckpt


def load_state_dict(ckpt):
    def get_state_dict_from_lightning(path):
        pl_sd = torch.load(path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        return sd

    print(f"Loading model from {ckpt}")
    if ckpt.endswith("ckpt"):
        if os.path.isdir(ckpt) and os.path.exists(
            os.path.join(ckpt, "pytorch_model.bin")
        ):
            sd = torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu")
        elif os.path.isdir(ckpt):
            # convert deepspeed checkpoint to fp32 state dict
            import tempfile
            from lightning.pytorch.utilities.deepspeed import (
                convert_zero_checkpoint_to_fp32_state_dict,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                fp32_ckpt = os.path.join(tmpdir, "pytorch_model.bin")
                convert_zero_checkpoint_to_fp32_state_dict(ckpt, fp32_ckpt)
                sd = get_state_dict_from_lightning(fp32_ckpt)
        else:
            sd = get_state_dict_from_lightning(ckpt)
    elif ckpt.endswith("safetensors"):
        sd = load_safetensors(ckpt)
    else:
        raise NotImplementedError
    return sd


def load_hf_model_from_config(
    model_path,
    ckpt,
    model_name="student_model",
    vocab_size=None,
):
    if vocab_size is not None:
        # load config
        model_config = AutoConfig.from_pretrained(model_path)
        model_config.vocab_size = vocab_size
    sd = load_state_dict(ckpt)
    sd = {k.replace(f"{model_name}.", ""): v for k, v in sd.items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        state_dict=sd,
    )
    return model
