import os
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from dataset.kttiDepthData import KttiDepthDataModule
from marigold.marigoldDepth import MarigoldDepth


def parse_args():
    parser = argparse.ArgumentParser("Train Marigold Depth")
    # Data
    parser.add_argument("--batch_size", type=int, default=4, help="Per-step batch size")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader workers")
    parser.add_argument("--image_size", type=int, default=512, help="H=W transform resize")
    # Optim/Train
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--precision", type=str, default="32", choices=["16", "16-mixed", "32"])
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    # Scheduler (ReduceLROnPlateau is configured inside the module)
    # System
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--val_check_interval", type=float, default=1, help="Validate N times per epoch (0<val<=1)")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    # Inference-time knobs (affect validation predict if you wire them in)
    parser.add_argument("--denoise_steps_val", type=int, default=10, help="Denoise steps for validation sanity checks")
    return parser.parse_args()


def build_data(batch_size: int, num_workers: int, image_size: int):
    # Ensure your Csv2ImageDepthDataset is resizing to image_size
    # If not, modify KttiDepthDataModule to accept image_size and pass it to transforms.Resize((image_size, image_size))
    data = KttiDepthDataModule(batch_size=batch_size, num_workers=num_workers)
    # Optionally you can set attributes if your module supports dynamic resize
    # data.set_image_size(image_size)
    return data


def build_model(learning_rate: float):
    torch.set_float32_matmul_precision('medium')
    model = MarigoldDepth(
        learning_rate=learning_rate,
        device_type="cuda",
        number_of_training_steps=40000,
    )
    return model


def build_trainer(
    max_epochs: int,
    devices: int,
    accelerator: str,
    precision: str,
    accumulate_grad_batches: int,
    gradient_clip_val: float,
    log_dir: str,
    ckpt_dir: str,
    check_val_every_n_epoch: int = 1,
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="marigold-depth-{epoch:02d}-{val_loss:.5f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=log_dir, name="marigold_depth")

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        log_every_n_steps=100,
        limit_train_batches=0.15,
        limit_val_batches=0.5,
    )
    return trainer


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    data = build_data(args.batch_size, args.num_workers, args.image_size)
    model = build_model(args.learning_rate)
    trainer = build_trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
    )

    # Optionally resume
    if args.resume_from:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
