import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from src.data import StreamingSFTDataModule
from src.model import KDForLM
from src.arguments import parse_args
from src.xpu_accelerator import XPUAccelerator

try:
    from lightning.pytorch.strategies import DeepSpeedStrategy, SingleDeviceStrategy
except ImportError:
    DeepSpeedStrategy = None
    SingleDeviceStrategy = None


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
    if args.accelerator == "xpu":
        if SingleDeviceStrategy is None:
            raise ImportError("Lightning SingleDeviceStrategy is unavailable.")
        devices = parse_devices(args.devices)
        if devices not in {"auto", 1}:
            raise ValueError("XPU execution currently supports only a single device.")
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU accelerator requested but torch.xpu is unavailable.")
        return SingleDeviceStrategy(device=torch.device("xpu", 0))

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


def get_accelerator(args):
    if args.accelerator == "xpu":
        return XPUAccelerator()
    return args.accelerator

if __name__ == "__main__":
    args = parse_args()
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    data = StreamingSFTDataModule(
        tokenizer_path=args.processor_model
        or args.tokenizer_model
        or args.teacher_model,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        trust_remote_code=args.trust_remote_code,
        use_processor=args.use_processor,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
    )
    model = KDForLM(args, preprocessor=data.preprocessor)
    modelcheckpoint = ModelCheckpoint(
        dirpath=args.output_dir,
        monitor="val_0/rougeL",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    trainer = L.Trainer(
        accelerator=get_accelerator(args),
        devices=parse_devices(args.devices),
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
        precision=args.precision,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        strategy=get_strategy(args),
        callbacks=[modelcheckpoint],
    )
    if args.validate_first:
        trainer.validate(model, data)
    trainer.fit(model, data)
