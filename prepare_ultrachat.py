import argparse
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from litdata import optimize
from functools import partial

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


def get_message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        )
    raise TypeError(f"Unsupported content type: {type(content)}")


def is_gpt_oss_tokenizer(tokenizer):
    name = str(getattr(tokenizer, "name_or_path", "")).lower()
    if "gpt-oss" in name:
        return True
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    return "<|start|>assistant" in chat_template and "Reasoning:" in chat_template


def build_chat_template_kwargs(tokenizer, add_generation_prompt, reasoning_effort):
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if is_gpt_oss_tokenizer(tokenizer) and reasoning_effort != "auto":
        template_kwargs["reasoning_effort"] = reasoning_effort
    return template_kwargs


def prepare_messages_for_template(messages, tokenizer):
    if not is_gpt_oss_tokenizer(tokenizer):
        return messages

    prefix = []
    index = 0
    while index < len(messages) and messages[index]["role"] in {"system", "developer"}:
        prefix.append(messages[index])
        index += 1

    if not prefix:
        return messages

    merged_content = "\n\n".join(
        text
        for text in (get_message_text(message["content"]) for message in prefix)
        if text
    )
    if not merged_content:
        return messages[index:]

    return [{"role": "developer", "content": merged_content}, *messages[index:]]


def build_chat_pair(messages, tokenizer, reasoning_effort):
    messages = prepare_messages_for_template(messages, tokenizer)
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message must be from the assistant.")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        **build_chat_template_kwargs(tokenizer, True, reasoning_effort),
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        **build_chat_template_kwargs(tokenizer, False, reasoning_effort),
    )
    return prompt_text, full_text, get_message_text(messages[-1]["content"])


def fits_length(example, tokenizer, max_input_len, max_output_len):
    max_length = max_input_len + max_output_len
    if len(example["model_inputs"]["input_ids"]) > max_length:
        return False
    if len(example["model_inputs_gen"]["input_ids"]) > max_input_len:
        return False
    output_tokens = tokenizer(example["response"]).input_ids
    if len(output_tokens) > max_output_len:
        return False
    return True


def build_training_example(messages, tokenizer, reasoning_effort):
    input_text, text, output_text = build_chat_pair(
        messages,
        tokenizer,
        reasoning_effort,
    )
    input_ids = tokenizer(text).input_ids
    gen_input_ids = tokenizer(input_text).input_ids
    return {
        "model_inputs": {"input_ids": input_ids, "labels": list(input_ids)},
        "model_inputs_gen": {"input_ids": gen_input_ids},
        "response": output_text,
    }


def truncate_prompt_window(example, tokenizer, max_input_len, max_output_len):
    output_ids = tokenizer(example["response"]).input_ids
    if len(output_ids) > max_output_len:
        return None

    prompt_ids = example["model_inputs_gen"]["input_ids"][-max_input_len:]
    full_ids = example["model_inputs"]["input_ids"]
    if output_ids and full_ids[-len(output_ids) :] == output_ids:
        full_ids = prompt_ids + output_ids
    else:
        full_ids = full_ids[-(len(prompt_ids) + len(output_ids)) :]

    return {
        "model_inputs": {"input_ids": full_ids, "labels": list(full_ids)},
        "model_inputs_gen": {"input_ids": prompt_ids},
        "response": example["response"],
    }


def iter_windowed_examples(
    messages, tokenizer, max_input_len, max_output_len, reasoning_effort
):
    if not messages:
        return []

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
                tokenizer,
                reasoning_effort,
            )
            if fits_length(example, tokenizer, max_input_len, max_output_len):
                samples.append(example)
                matched = True
                break

        if matched:
            continue

        window_messages = prefix + body[start_candidates[-1] : end_idx + 1]
        example = build_training_example(window_messages, tokenizer, reasoning_effort)
        example = truncate_prompt_window(
            example, tokenizer, max_input_len, max_output_len
        )
        if example is not None:
            samples.append(example)

    return samples


def tokenize_batch(batch, tokenizer, max_input_len, max_output_len, reasoning_effort):
    column = "messages" if "messages" in batch else "chosen"
    results = {"model_inputs": [], "model_inputs_gen": [], "response": []}

    for messages in batch[column]:
        for example in iter_windowed_examples(
            messages,
            tokenizer,
            max_input_len,
            max_output_len,
            reasoning_effort,
        ):
            results["model_inputs"].append(example["model_inputs"])
            results["model_inputs_gen"].append(example["model_inputs_gen"])
            results["response"].append(example["response"])

    return results


def fn(index, data):
    yield data[index]


def resolve_num_proc(requested_num_proc):
    if os.name == "nt":
        return min(requested_num_proc, 32)
    return requested_num_proc


def prepare_train(args, tokenizer):
    max_input_len = args.max_length - args.max_output_length
    num_proc = resolve_num_proc(args.num_proc)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=32,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_len": max_input_len,
            "max_output_len": args.max_output_length,
            "reasoning_effort": args.reasoning_effort,
        },
        num_proc=num_proc,
        desc="Applying chat templates",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "train"),
        num_workers=16,
        chunk_bytes="500MB",
    )


def prepare_test(args, tokenizer):
    max_input_len = args.max_length - args.max_output_length
    num_proc = resolve_num_proc(args.num_proc)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=32,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_len": max_input_len,
            "max_output_len": args.max_output_length,
            "reasoning_effort": args.reasoning_effort,
        },
        num_proc=num_proc,
        desc="Applying chat templates",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    ds = dataset.train_test_split(test_size=2000, seed=42, shuffle=True)
    dataset = ds["test"]

    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "test"),
        num_workers=2,
        chunk_bytes="500MB",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODELS.keys()),
        default="phi-3",
        help="Teacher type",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="optional tokenizer/model path override for chat templating",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--num_proc", type=int, default=64, help="number of workers for processing"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=MAX_LENGTH,
        help="maximum total sequence length after chat templating",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=MAX_OUTPUT_LENGTH,
        help="maximum assistant response length",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="allow transformers to load custom tokenizer code",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="auto",
        choices=["auto", "low", "medium", "high"],
        help="reasoning_effort to embed in GPT-OSS harmony chat templates",
    )
    args = parser.parse_args()
    tokenizer_path = args.tokenizer_name or MODELS[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    if args.model_type == "phi-3":
        # https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/sample_finetune.py#L141
        tokenizer.pad_token = (
            tokenizer.unk_token
        )  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.max_output_length >= args.max_length:
        raise ValueError("--max_output_length must be smaller than --max_length.")
    prepare_train(args, tokenizer)
    prepare_test(args, tokenizer)
