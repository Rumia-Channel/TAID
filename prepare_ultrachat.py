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


def build_chat_pair(messages, tokenizer):
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message must be from the assistant.")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text, get_message_text(messages[-1]["content"])


def tokenize(example, tokenizer):
    column = "messages" if "messages" in example else "chosen"
    input_text, text, output_text = build_chat_pair(example[column], tokenizer)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    res = {"model_inputs": {"input_ids": input_ids, "labels": input_ids.clone()}}

    gen_input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    res["model_inputs_gen"] = {"input_ids": gen_input_ids}
    res["response"] = output_text
    return res


def filter_length(example, tokenizer, max_input_len, max_output_len):
    max_length = max_input_len + max_output_len
    if example["model_inputs"]["input_ids"].size(1) > max_length:
        return False
    if example["model_inputs_gen"]["input_ids"].size(1) > max_input_len:
        return False
    output_tokens = tokenizer(example["response"], return_tensors="pt").input_ids
    if output_tokens.size(1) > max_output_len:
        return False
    return True


def fn(index, data):
    yield data[index]


def prepare_train(args, tokenizer):
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "train"),
        num_workers=16,
        chunk_bytes="500MB",
    )


def prepare_test(args, tokenizer):
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
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
        "--trust_remote_code",
        action="store_true",
        help="allow transformers to load custom tokenizer code",
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
    prepare_train(args, tokenizer)
    prepare_test(args, tokenizer)
