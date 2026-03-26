import os
from typing import Optional

import torch
import lightning as L
from litdata import StreamingDataset, StreamingDataLoader
from src.utils import load_preprocessor, get_text_tokenizer, get_pad_token_id

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _pad(
    inputs,
    key: str,
    max_length: int,
    pad_id: int,
    padding_side: str,
):
    assert padding_side in ["right", "left"]
    if padding_side == "right":
        if isinstance(inputs[0][key], torch.Tensor):
            # assume that each tensor is not truncated yet and the shape is (1, seq_len)
            data = torch.stack(
                [
                    torch.cat(
                        [
                            d[key][0][:max_length],
                            torch.tensor(
                                [pad_id]
                                * (max_length - d[key][0][:max_length].size(0)),
                                dtype=torch.long,
                            ),
                        ]
                    )
                    for d in inputs
                ]
            )
        else:
            data = [d[key] + [pad_id] * (max_length - len(d[key])) for d in inputs]
    else:
        if isinstance(inputs[0][key], torch.Tensor):
            data = torch.stack(
                [
                    torch.cat(
                        [
                            torch.tensor(
                                [pad_id]
                                * (max_length - d[key][0][:max_length].size(0)),
                                dtype=torch.long,
                            ),
                            d[key][0][:max_length],
                        ]
                    )
                    for d in inputs
                ]
            )
        else:
            data = [[pad_id] * (max_length - len(d[key])) + d[key] for d in inputs]
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)
    return data


class StreamingDataCollatorForLM:
    def __init__(self, preprocessor, max_input_len, max_output_len):
        self.preprocessor = preprocessor
        self.tokenizer = get_text_tokenizer(preprocessor)
        self.max_input_len = max_input_len
        self.max_length = max_input_len + max_output_len

    def pad(self, inputs, max_length: int, padding_side: str = "right"):
        data = {}
        keys = inputs[0].keys()
        for k in keys:
            if k == "input_ids":
                pad_id = get_pad_token_id(self.preprocessor)
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k == "attention_mask":
                pad_id = 0
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k == "mm_token_type_ids":
                pad_id = 0
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k == "labels":
                pad_id = -100
                data[k] = _pad(inputs, k, max_length, pad_id, padding_side)
            elif k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]:
                data[k] = torch.cat([d[k] for d in inputs], dim=0)
            else:
                data[k] = [d[k] for d in inputs]
        if "attention_mask" not in data:
            data["attention_mask"] = (
                data["input_ids"].ne(get_pad_token_id(self.preprocessor)).long()
            )
        return data

    def __call__(self, samples):
        result = {}
        model_inputs_columns = [
            col for col in list(samples[0].keys()) if col.startswith("model_inputs")
        ]
        for col in model_inputs_columns:
            # list of dict
            inputs = [sample[col] for sample in samples]
            if col == "model_inputs_gen":
                result[col] = self.pad(inputs, self.max_input_len, padding_side="left")
            else:
                result[col] = self.pad(inputs, self.max_length, padding_side="right")
        if "model_inputs_gen" in result:
            result["model_inputs_gen"]["response"] = [s["response"] for s in samples]
        return result


class StreamingSFTDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer_path: str,
        data_path: str,
        batch_size: int,
        num_workers: int,
        trust_remote_code: bool = False,
        use_processor: bool = False,
        eval_batch_size: Optional[int] = None,
        max_input_len: int = 1536,
        max_output_len: int = 512,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size else batch_size
        self.num_workers = num_workers
        self.preprocessor = load_preprocessor(
            tokenizer_path,
            use_processor=use_processor,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = get_text_tokenizer(self.preprocessor)

        self.collate_fn = StreamingDataCollatorForLM(
            preprocessor=self.preprocessor,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
        )
        self.loader_cls = StreamingDataLoader
        self.datasets = {}

    def setup(self, stage: str):
        self.datasets["train"] = StreamingDataset(os.path.join(self.data_path, "train"))
        self.datasets["val"] = StreamingDataset(os.path.join(self.data_path, "test"))

    def train_dataloader(self):
        train_dataloader_kwargs = {
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
        }
        return self.loader_cls(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            **train_dataloader_kwargs,
        )

    def val_dataloader(self):
        return self.loader_cls(
            self.datasets["val"],
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
