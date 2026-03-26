import argparse
import os
import re
from functools import partial

import torch
from datasets import load_dataset
from litdata import optimize

from src.utils import (
    load_preprocessor,
    get_text_tokenizer,
    is_gpt_oss_preprocessor,
)

MODELS = {
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "stablelm": "stabilityai/stablelm-zephyr-3b",
    "qwen3.5": "Qwen/Qwen3.5-9B",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}
MAX_LENGTH = 2048
MAX_OUTPUT_LENGTH = 512
DEFAULT_EVAL_SIZE = 2000
DEFAULT_EVAL_RATIO = 0.02
MODALITY_PLACEHOLDERS = {
    "image": "[IMAGE]",
    "video": "[VIDEO]",
    "audio": "[AUDIO]",
    "document": "[DOCUMENT]",
}
ROLE_ALIASES = {
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "tool": "tool",
    "function": "tool",
    "system": "system",
    "developer": "developer",
    "user": "user",
    "human": "user",
}


def parse_csv_arg(value):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    return [part for part in parts if part]


def strip_thinking_sections(text: str) -> str:
    return re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL)


def maybe_strip_thinking_text(text: str, enable_thinking: str) -> str:
    if enable_thinking == "false":
        return strip_thinking_sections(text).strip()
    return text


def canonicalize_role(raw_role, preserve_developer=False):
    role = str(raw_role).strip().lower()
    if role not in ROLE_ALIASES:
        raise ValueError(f"Unsupported role: {raw_role}")
    if role == "developer" and not preserve_developer:
        return "system"
    return ROLE_ALIASES[role]


def supports_developer_role(args):
    selected = str(args.tokenizer_name or args.model_type or "").lower()
    return "gpt-oss" in selected


def infer_modality(item):
    item_type = str(item.get("type", "")).strip().lower()
    if item_type:
        return item_type
    if "image" in item or "image_url" in item:
        return "image"
    if "video" in item or "video_url" in item:
        return "video"
    if "audio" in item or "audio_url" in item:
        return "audio"
    if "document" in item or "file" in item:
        return "document"
    if "text" in item:
        return "text"
    return "unknown"


def placeholder_for_modality(modality):
    base = modality.split(".", 1)[0]
    return MODALITY_PLACEHOLDERS.get(base, f"[{base.upper()}]")


def normalize_content_for_template(content, multimodal_mode, use_processor=False):
    if content is None:
        return [] if use_processor and multimodal_mode == "preserve" else ""
    if isinstance(content, str):
        if use_processor and multimodal_mode == "preserve":
            return [{"type": "text", "text": content}]
        return content
    if not isinstance(content, list):
        content = str(content)
        if use_processor and multimodal_mode == "preserve":
            return [{"type": "text", "text": content}]
        return content

    if multimodal_mode == "preserve":
        normalized = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                raise TypeError(f"Unsupported content item type: {type(item)}")
            modality = infer_modality(item)
            if modality == "text":
                normalized.append({"type": "text", "text": str(item.get("text", ""))})
                continue
            if modality == "unknown":
                raise ValueError(f"Unsupported multimodal content item: {item}")
            normalized.append(dict(item))
        return normalized

    text_parts = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Unsupported content item type: {type(item)}")
        modality = infer_modality(item)
        if modality == "text":
            text_parts.append(str(item.get("text", "")))
            continue
        if multimodal_mode == "strict":
            raise ValueError(f"Encountered multimodal content in strict mode: {item}")
        if multimodal_mode == "placeholder":
            text_parts.append(placeholder_for_modality(modality))
    return "".join(text_parts)


def render_content_text(content, multimodal_mode, enable_thinking, use_processor=False):
    if isinstance(content, str):
        return maybe_strip_thinking_text(content, enable_thinking)
    rendered = normalize_content_for_template(
        content,
        multimodal_mode,
        use_processor=use_processor,
    )
    if isinstance(rendered, list):
        text_parts = []
        for item in rendered:
            modality = infer_modality(item)
            if modality == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(placeholder_for_modality(modality))
        rendered = "".join(text_parts)
    return maybe_strip_thinking_text(str(rendered), enable_thinking)


def apply_chat_template(
    preprocessor,
    messages,
    add_generation_prompt,
    enable_thinking,
    reasoning_effort,
):
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if is_gpt_oss_preprocessor(preprocessor):
        if reasoning_effort != "auto":
            template_kwargs["reasoning_effort"] = reasoning_effort
    elif enable_thinking != "auto":
        template_kwargs["enable_thinking"] = enable_thinking == "true"
    rendered = preprocessor.apply_chat_template(messages, **template_kwargs)
    return maybe_strip_thinking_text(rendered, enable_thinking)


def collect_visual_inputs(messages):
    images = []
    videos = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            modality = infer_modality(item)
            if modality.startswith("image"):
                image = item.get("image") or item.get("image_url")
                if image is not None:
                    images.append(image)
            elif modality.startswith("video"):
                video = item.get("video") or item.get("video_url")
                if video is not None:
                    videos.append(video)
    return images or None, videos or None


def prepare_messages_for_template(
    messages,
    preprocessor,
    multimodal_mode,
    enable_thinking,
):
    if not is_gpt_oss_preprocessor(preprocessor):
        return messages

    prefix = []
    index = 0
    while index < len(messages) and messages[index]["role"] in {"system", "developer"}:
        prefix.append(messages[index])
        index += 1

    if not prefix:
        return messages

    content_parts = []
    for message in prefix:
        rendered = render_content_text(
            message["content"],
            multimodal_mode,
            enable_thinking,
            use_processor=hasattr(preprocessor, "tokenizer"),
        )
        if rendered:
            content_parts.append(rendered)
    merged_content = "\n\n".join(content_parts)
    if not merged_content:
        return messages[index:]

    return [{"role": "developer", "content": merged_content}, *messages[index:]]


def build_chat_pair(
    messages,
    preprocessor,
    multimodal_mode,
    enable_thinking,
    reasoning_effort,
):
    messages = prepare_messages_for_template(
        messages,
        preprocessor,
        multimodal_mode,
        enable_thinking,
    )
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message must be from the assistant.")

    prompt_text = apply_chat_template(
        preprocessor,
        messages[:-1],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    full_text = apply_chat_template(
        preprocessor,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    response_text = render_content_text(
        messages[-1]["content"],
        multimodal_mode,
        enable_thinking,
        use_processor=hasattr(preprocessor, "tokenizer"),
    )
    return prompt_text, full_text, response_text


def sequence_length(value):
    if hasattr(value, "size"):
        return value.size(-1)
    return len(value)


def fits_length(example, tokenizer, max_input_len, max_output_len):
    max_length = max_input_len + max_output_len
    if sequence_length(example["model_inputs"]["input_ids"]) > max_length:
        return False
    if sequence_length(example["model_inputs_gen"]["input_ids"]) > max_input_len:
        return False
    output_tokens = tokenizer(example["response"]).input_ids
    if len(output_tokens) > max_output_len:
        return False
    return True


def build_training_example(
    messages,
    preprocessor,
    multimodal_mode,
    enable_thinking,
    reasoning_effort,
):
    input_text, text, output_text = build_chat_pair(
        messages,
        preprocessor,
        multimodal_mode=multimodal_mode,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
    )
    tokenizer = get_text_tokenizer(preprocessor)
    if hasattr(preprocessor, "tokenizer"):
        prompt_images, prompt_videos = collect_visual_inputs(messages[:-1])
        full_images, full_videos = collect_visual_inputs(messages)
        model_inputs = preprocessor(
            text=[text],
            images=full_images,
            videos=full_videos,
            return_tensors="pt",
            padding=True,
        )
        model_inputs_gen = preprocessor(
            text=[input_text],
            images=prompt_images,
            videos=prompt_videos,
            return_tensors="pt",
            padding=True,
        )
    else:
        input_ids = tokenizer(text).input_ids
        gen_input_ids = tokenizer(input_text).input_ids
        model_inputs = {"input_ids": input_ids}
        model_inputs_gen = {"input_ids": gen_input_ids}

    if "input_ids" in model_inputs and hasattr(model_inputs["input_ids"], "clone"):
        model_inputs["labels"] = model_inputs["input_ids"].clone()
    else:
        model_inputs["labels"] = list(model_inputs["input_ids"])
    return {
        "model_inputs": model_inputs,
        "model_inputs_gen": model_inputs_gen,
        "response": output_text,
    }


def truncate_prompt_window(example, tokenizer, max_input_len, max_output_len):
    output_ids = tokenizer(example["response"]).input_ids
    if len(output_ids) > max_output_len:
        return None

    prompt_ids = example["model_inputs_gen"]["input_ids"]
    full_ids = example["model_inputs"]["input_ids"]
    if hasattr(prompt_ids, "size"):
        prompt_ids = prompt_ids[:, -max_input_len:]
        output_tensor = full_ids.new_tensor(output_ids)
        if output_ids and torch.equal(full_ids[0, -len(output_ids) :], output_tensor):
            full_ids = torch.cat([prompt_ids, output_tensor.unsqueeze(0)], dim=-1)
        else:
            full_ids = full_ids[:, -(prompt_ids.size(-1) + len(output_ids)) :]
        model_inputs = {}
        for key, value in example["model_inputs"].items():
            if key == "input_ids":
                model_inputs[key] = full_ids
            elif key == "labels":
                model_inputs[key] = full_ids.clone()
            elif key in ["attention_mask", "mm_token_type_ids"]:
                tail = prompt_ids.size(-1) + len(output_ids)
                model_inputs[key] = value[:, -tail:]
                if key == "attention_mask":
                    model_inputs[key] = full_ids.ne(tokenizer.pad_token_id).long()
            else:
                model_inputs[key] = value
        model_inputs_gen = {}
        for key, value in example["model_inputs_gen"].items():
            if key in ["input_ids", "attention_mask", "mm_token_type_ids"]:
                model_inputs_gen[key] = value[:, -prompt_ids.size(-1) :]
                if key == "attention_mask":
                    model_inputs_gen[key] = prompt_ids.ne(tokenizer.pad_token_id).long()
            else:
                model_inputs_gen[key] = value
    else:
        prompt_ids = prompt_ids[-max_input_len:]
        if output_ids and full_ids[-len(output_ids) :] == output_ids:
            full_ids = prompt_ids + output_ids
        else:
            full_ids = full_ids[-(len(prompt_ids) + len(output_ids)) :]
        model_inputs = {"input_ids": full_ids, "labels": list(full_ids)}
        model_inputs_gen = {"input_ids": prompt_ids}

    return {
        "model_inputs": model_inputs,
        "model_inputs_gen": model_inputs_gen,
        "response": example["response"],
    }


def iter_windowed_examples(
    messages,
    preprocessor,
    max_input_len,
    max_output_len,
    multimodal_mode,
    enable_thinking,
    reasoning_effort,
):
    if not messages:
        return []

    tokenizer = get_text_tokenizer(preprocessor)
    prefix = []
    for message in messages:
        if message["role"] not in {"system", "developer"}:
            break
        prefix.append(message)
    body = messages[len(prefix) :]
    samples = []

    for end_idx, message in enumerate(body):
        if message["role"] != "assistant":
            continue

        start_candidates = [0]
        for idx, previous in enumerate(body[:end_idx]):
            if previous["role"] == "assistant":
                start_candidates.append(idx + 1)

        matched = False
        for start_idx in start_candidates:
            window_messages = prefix + body[start_idx : end_idx + 1]
            example = build_training_example(
                window_messages,
                preprocessor,
                multimodal_mode=multimodal_mode,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
            if fits_length(example, tokenizer, max_input_len, max_output_len):
                samples.append(example)
                matched = True
                break

        if matched:
            continue

        window_messages = prefix + body[start_candidates[-1] : end_idx + 1]
        example = build_training_example(
            window_messages,
            preprocessor,
            multimodal_mode=multimodal_mode,
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        example = truncate_prompt_window(
            example, tokenizer, max_input_len, max_output_len
        )
        if example is not None:
            samples.append(example)

    return samples


def detect_input_format(example, args):
    if args.input_format != "auto":
        return args.input_format

    if args.messages_column and args.messages_column in example:
        return "messages"
    if args.chosen_column and args.chosen_column in example:
        return "chosen"
    if args.conversations_column and args.conversations_column in example:
        return "sharegpt"
    if args.output_column and args.output_column in example:
        return "alpaca"
    if args.response_column and args.response_column in example:
        return "prompt-response"

    if "messages" in example:
        return "messages"
    if "chosen" in example:
        return "chosen"
    if "conversations" in example or "conversation" in example:
        return "sharegpt"
    if "instruction" in example and "output" in example:
        return "alpaca"
    if "prompt" in example and ("response" in example or "completion" in example):
        return "prompt-response"
    if "question" in example and ("answer" in example or "response" in example):
        return "prompt-response"

    raise ValueError(
        "Could not infer input format. Provide --input_format and matching column options."
    )


def normalize_message_list(messages, args, role_field, content_field):
    normalized = []
    preserve_developer = supports_developer_role(args)
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"Unsupported message type: {type(message)}")
        normalized.append(
            {
                "role": canonicalize_role(
                    message[role_field],
                    preserve_developer=preserve_developer,
                ),
                "content": normalize_content_for_template(
                    message.get(content_field),
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
    return normalized


def normalize_sharegpt(messages, args):
    normalized = []
    preserve_developer = supports_developer_role(args)
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError(f"Unsupported message type: {type(message)}")
        normalized.append(
            {
                "role": canonicalize_role(
                    message[args.sharegpt_role_field],
                    preserve_developer=preserve_developer,
                ),
                "content": normalize_content_for_template(
                    message.get(args.sharegpt_content_field),
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
    return normalized


def build_prompt_text(prompt, input_text):
    prompt = "" if prompt is None else str(prompt)
    input_text = "" if input_text is None else str(input_text)
    if prompt and input_text:
        return f"{prompt}\n\n{input_text}"
    return prompt or input_text


def normalize_example(example, args):
    input_format = detect_input_format(example, args)
    if input_format == "messages":
        column = args.messages_column or "messages"
        return normalize_message_list(
            example[column],
            args,
            role_field=args.role_field,
            content_field=args.content_field,
        )
    if input_format == "chosen":
        column = args.chosen_column or "chosen"
        return normalize_message_list(
            example[column],
            args,
            role_field=args.role_field,
            content_field=args.content_field,
        )
    if input_format == "sharegpt":
        column = args.conversations_column or (
            "conversations" if "conversations" in example else "conversation"
        )
        return normalize_sharegpt(example[column], args)
    if input_format == "alpaca":
        messages = []
        if args.system_column and example.get(args.system_column):
            messages.append(
                {
                    "role": "system",
                    "content": normalize_content_for_template(
                        example.get(args.system_column),
                        args.multimodal_mode,
                        use_processor=args.use_processor,
                    ),
                }
            )
        instruction_column = args.instruction_column or "instruction"
        input_column = args.input_column or "input"
        output_column = args.output_column or "output"
        user_text = build_prompt_text(
            example.get(instruction_column),
            example.get(input_column),
        )
        messages.append(
            {
                "role": "user",
                "content": normalize_content_for_template(
                    user_text,
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": normalize_content_for_template(
                    example.get(output_column),
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
        return messages
    if input_format == "prompt-response":
        prompt_column = args.prompt_column or (
            "prompt" if "prompt" in example else "question"
        )
        response_column = args.response_column or (
            "response"
            if "response" in example
            else ("completion" if "completion" in example else "answer")
        )
        messages = []
        if args.system_column and example.get(args.system_column):
            messages.append(
                {
                    "role": "system",
                    "content": normalize_content_for_template(
                        example.get(args.system_column),
                        args.multimodal_mode,
                        use_processor=args.use_processor,
                    ),
                }
            )
        messages.append(
            {
                "role": "user",
                "content": normalize_content_for_template(
                    example.get(prompt_column),
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": normalize_content_for_template(
                    example.get(response_column),
                    args.multimodal_mode,
                    use_processor=args.use_processor,
                ),
            }
        )
        return messages
    raise ValueError(f"Unsupported input format: {input_format}")


def tokenize_batch(batch, preprocessor, args):
    batch_size = len(next(iter(batch.values()))) if batch else 0
    results = {"model_inputs": [], "model_inputs_gen": [], "response": []}
    max_input_len = args.max_length - args.max_output_length

    for index in range(batch_size):
        example = {key: value[index] for key, value in batch.items()}
        messages = normalize_example(example, args)
        for sample in iter_windowed_examples(
            messages,
            preprocessor,
            max_input_len=max_input_len,
            max_output_len=args.max_output_length,
            multimodal_mode=args.multimodal_mode,
            enable_thinking=args.enable_thinking,
            reasoning_effort=args.reasoning_effort,
        ):
            results["model_inputs"].append(sample["model_inputs"])
            results["model_inputs_gen"].append(sample["model_inputs_gen"])
            results["response"].append(sample["response"])
    return results


def fn(index, data):
    yield data[index]


def resolve_num_proc(requested_num_proc):
    if os.name == "nt":
        return min(requested_num_proc, 32)
    return requested_num_proc


def derive_output_name(args):
    if args.output_name:
        return args.output_name
    dataset_name = args.dataset.replace("\\", "/").rstrip("/").split("/")[-1]
    dataset_name = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset_name)
    if dataset_name and args.model_type:
        return f"{dataset_name}-{args.model_type}"
    if dataset_name:
        return dataset_name
    if args.model_type:
        return args.model_type
    return "prepared-data"


def load_source_dataset(args, split_name):
    load_kwargs = {}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    if args.data_dir:
        load_kwargs["data_dir"] = args.data_dir
    data_files = {}
    if args.train_files:
        data_files["train"] = args.train_files
    if args.eval_files:
        data_files["validation"] = args.eval_files
    if data_files:
        load_kwargs["data_files"] = data_files
    return load_dataset(args.dataset, split=split_name, **load_kwargs)


def resolve_eval_size(length, args):
    if length < 2:
        raise ValueError("Need at least 2 training examples to derive an eval split.")
    eval_size = min(args.eval_size, max(1, length - 1))
    if eval_size > 0:
        return eval_size
    ratio_size = max(1, int(length * args.eval_ratio))
    return min(ratio_size, max(1, length - 1))


def prepare_split(dataset, preprocessor, args, output_subdir, num_workers):
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.map_batch_size,
        fn_kwargs={"preprocessor": preprocessor, "args": args},
        num_proc=resolve_num_proc(args.num_proc),
        desc=f"Applying chat templates ({output_subdir})",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, derive_output_name(args), output_subdir),
        num_workers=num_workers,
        chunk_bytes=args.chunk_bytes,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset name/path for datasets.load_dataset, or loader name such as json/parquet",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="optional dataset config name",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="train split name when reading from a dataset repository",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="optional eval split name when reading from a dataset repository",
    )
    parser.add_argument(
        "--train_files",
        type=parse_csv_arg,
        default=None,
        help="comma-separated files for the train split when using loaders like json/parquet",
    )
    parser.add_argument(
        "--eval_files",
        type=parse_csv_arg,
        default=None,
        help="comma-separated files for the eval split when using loaders like json/parquet",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="auto",
        choices=["auto", "messages", "chosen", "sharegpt", "alpaca", "prompt-response"],
    )
    parser.add_argument("--messages_column", type=str, default=None)
    parser.add_argument("--chosen_column", type=str, default=None)
    parser.add_argument("--conversations_column", type=str, default=None)
    parser.add_argument("--role_field", type=str, default="role")
    parser.add_argument("--content_field", type=str, default="content")
    parser.add_argument("--sharegpt_role_field", type=str, default="from")
    parser.add_argument("--sharegpt_content_field", type=str, default="value")
    parser.add_argument("--system_column", type=str, default=None)
    parser.add_argument("--instruction_column", type=str, default=None)
    parser.add_argument("--input_column", type=str, default=None)
    parser.add_argument("--output_column", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default=None)
    parser.add_argument("--response_column", type=str, default=None)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODELS.keys()),
        default=None,
        help="named tokenizer preset",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="tokenizer/model path override used for chat templating",
    )
    parser.add_argument(
        "--use_processor",
        action="store_true",
        help="use AutoProcessor for native multimodal models",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="subdirectory name under output_dir",
    )
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--map_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max_output_length", type=int, default=MAX_OUTPUT_LENGTH)
    parser.add_argument(
        "--multimodal_mode",
        type=str,
        default="preserve",
        choices=["preserve", "placeholder", "text-only", "strict"],
        help="how to normalize multimodal content before chat templating",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="pass enable_thinking to chat templates and strip think tags when false",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="auto",
        choices=["auto", "low", "medium", "high"],
        help="reasoning_effort to embed in GPT-OSS harmony chat templates",
    )
    parser.add_argument("--eval_size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--eval_ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_bytes", type=str, default="500MB")
    parser.add_argument("--train_optimize_workers", type=int, default=16)
    parser.add_argument("--eval_optimize_workers", type=int, default=2)
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="allow transformers to load custom tokenizer code",
    )
    args = parser.parse_args()
    if not args.model_type and not args.tokenizer_name:
        parser.error("Specify either --model_type or --tokenizer_name.")
    if args.max_output_length >= args.max_length:
        parser.error("--max_output_length must be smaller than --max_length.")
    return args


if __name__ == "__main__":
    args = parse_args()
    tokenizer_path = args.tokenizer_name or MODELS[args.model_type]
    preprocessor = load_preprocessor(
        tokenizer_path,
        use_processor=args.use_processor,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset = load_source_dataset(args, args.train_split)
    if args.eval_split or args.eval_files:
        eval_split_name = args.eval_split or "validation"
        eval_dataset = load_source_dataset(args, eval_split_name)
    else:
        eval_size = resolve_eval_size(len(train_dataset), args)
        split = train_dataset.train_test_split(
            test_size=eval_size,
            seed=args.seed,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    output_root = os.path.join(args.output_dir, derive_output_name(args))
    os.makedirs(output_root, exist_ok=True)
    prepare_split(
        train_dataset,
        preprocessor,
        args,
        output_subdir="train",
        num_workers=args.train_optimize_workers,
    )
    prepare_split(
        eval_dataset,
        preprocessor,
        args,
        output_subdir="test",
        num_workers=args.eval_optimize_workers,
    )
