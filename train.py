import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.data import StreamingSFTDataModule
from src.model import KDForLM
from src.arguments import parse_args

try:
    from lightning.pytorch.strategies import DeepSpeedStrategy
except ImportError:
    DeepSpeedStrategy = None


def parse_devices(value: str):
    if value == "auto":
        return value
    if "," in value:
        return value
    try:
        return int(value)
    except ValueError:
        return value


def get_strategy(args):
    if args.strategy == "deepspeed_stage_2":
        if DeepSpeedStrategy is None:
            raise ImportError(
                "deepspeed is not installed. Run `uv sync --extra cuda --extra deepspeed` first."
            )
        return DeepSpeedStrategy(
            stage=2, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        )

    if args.strategy != "auto":
        return args.strategy

    if (
        DeepSpeedStrategy is not None
        and args.accelerator in {"auto", "cuda"}
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        return DeepSpeedStrategy(
            stage=2, allgather_bucket_size=5e8, reduce_bucket_size=5e8
        )
    return "auto"

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    data = StreamingSFTDataModule(
        tokenizer_path=args.teacher_model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = KDForLM(args, tokenizer=data.tokenizer)
    modelcheckpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor="val_0/rougeL",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        precision=args.precision,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        limit_train_batches=10,
        limit_val_batches=5,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=get_strategy(args),
        callbacks=[modelcheckpoint],
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data)
