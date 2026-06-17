"""Fine-tuning loop for VITS (kakao-enterprise/vits-ljs) on a custom voice.

Pipeline: config + model_config -> dataset -> DataLoader(collate_fn) -> model
-> optimizer -> epoch/batch loop (forward_train -> losses -> backward -> step),
with periodic logging, checkpointing, and resume-from-checkpoint.

Run with:
    PYTHONPATH=src python -m vits_finetune.train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vits_finetune.checkpoint import load_checkpoint, save_checkpoint
from vits_finetune.collate import collate_fn
from vits_finetune.config import TrainingConfig
from vits_finetune.dataset import VitsFinetuneDataset
from vits_finetune.discriminator import VitsDiscriminator
from vits_finetune.losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
    kl_loss,
    recon_loss,
)
from vits_finetune.model import VitsFinetuneModel
from vits_finetune.model_config import VitsModelConfig

logger = logging.getLogger(__name__)

class Trainer:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """CLI overrides for the most commonly tuned ``TrainingConfig`` fields."""
        parser = argparse.ArgumentParser(description='Fine-tune VITS on a custom voice.')
        parser.add_argument('--dataset-repo-id', default=None, help='HF Hub dataset repo id.')
        parser.add_argument('--checkpoint-dir', type=Path, default=None)
        parser.add_argument('--batch-size', type=int, default=None)
        parser.add_argument('--learning-rate', type=float, default=None)
        parser.add_argument('--num-epochs', type=int, default=None)
        parser.add_argument('--num-workers', type=int, default=None, help='DataLoader workers.')
        parser.add_argument('--max-train-clips', type=int, default=None,
                            help='Cap training clips (Colab espeak-leak workaround); test set always held out.')
        parser.add_argument('--resume', type=Path, default=None, help='Checkpoint .pt to resume from.')
        parser.add_argument('--device', default=None, help='"cuda" or "cpu" (default: auto-detect).')
        return parser.parse_args()

    @staticmethod
    def build_config(args: argparse.Namespace) -> TrainingConfig:
        """Build a ``TrainingConfig``, applying any non-``None`` CLI overrides from ``args``."""
        overrides = {
            name: value
            for name in (
                'dataset_repo_id', 'checkpoint_dir',
                'batch_size', 'learning_rate', 'num_epochs',
                'num_workers', 'max_train_clips',
            )
            if (value := getattr(args, name, None)) is not None
        }
        return TrainingConfig(**overrides)

    def __init__(
        self,
        config: TrainingConfig,
        model_config: VitsModelConfig,
        args: argparse.Namespace,
    ) -> None:
        self.device = torch.device(
            args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        logger.info(f'Using device: {self.device}')

        torch.manual_seed(config.seed)
        self.config = config
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.dataset = VitsFinetuneDataset(config, self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
        )
        self.model = VitsFinetuneModel(model_config, config).to(self.device)
        self.discriminator = VitsDiscriminator().to(self.device)
        self.optim_g = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=(0.8, 0.99)
        )
        self.optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.disc_learning_rate,
            betas=(0.8, 0.99),
        )
        self.start_epoch, self.global_step = 0, 0

    def train_step(self, batch: dict) -> dict:
        """One GAN step: discriminator update, then generator update."""
        outputs = self.model.forward_train(batch)
        real = outputs['target_waveform']
        fake = outputs['predicted_waveform']

        # --- discriminator step ---
        self.optim_d.zero_grad()
        d_real, d_fake, _, _ = self.discriminator(real, fake.detach())
        loss_d = self.config.disc_loss_weight * discriminator_loss(d_real, d_fake)
        loss_d.backward()
        self.optim_d.step()

        # --- generator step ---
        self.optim_g.zero_grad()
        _, d_fake_g, fmap_real, fmap_fake = self.discriminator(real, fake)
        recon = recon_loss(outputs['predicted_mel'], outputs['target_mel'])
        kl = kl_loss(
            outputs['z_p'],
            outputs['posterior_log_stddev'],
            outputs['prior_mean'],
            outputs['prior_log_stddev'],
            outputs['z_mask'],
        )
        dur = outputs['duration_loss']
        adv = generator_adv_loss(d_fake_g)
        fm = feature_matching_loss(fmap_real, fmap_fake)
        loss_g = (
            self.config.mel_loss_weight * recon
            + self.config.kl_loss_weight * kl
            + dur
            + self.config.gen_loss_weight * adv
            + self.config.fm_loss_weight * fm
        )
        loss_g.backward()
        self.optim_g.step()

        return {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'recon': recon.item(),
            'kl': kl.item(),
            'dur': dur.item(),
            'adv': adv.item(),
            'fm': fm.item(),
        }

    def _save(self, name: str, epoch: int) -> None:
        path = self.config.checkpoint_dir / name
        save_checkpoint(
            path,
            self.model,
            self.optim_g,
            self.global_step,
            epoch,
            discriminator=self.discriminator,
            disc_optimizer=self.optim_d,
        )
        logger.info(f'Saved checkpoint to {path}')

    def train(self) -> None:
        """Run the GAN fine-tuning loop end to end, with resume and checkpointing."""
        if self.args.resume:
            ckpt = load_checkpoint(
                self.args.resume,
                self.model,
                self.optim_g,
                discriminator=self.discriminator,
                disc_optimizer=self.optim_d,
                map_location=self.device,
            )
            self.start_epoch = ckpt.get('epoch', 0)
            self.global_step = ckpt.get('step', 0)
            logger.info(
                f'Resumed from {self.args.resume} at epoch {self.start_epoch}, '
                f'step {self.global_step}'
            )

        for epoch in range(self.start_epoch, self.config.num_epochs):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss_dict = self.train_step(batch)

                if self.global_step % self.config.log_every == 0:
                    logger.info(
                        f'Epoch {epoch} | Step {self.global_step} | '
                        + ' | '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items())
                    )
                if self.global_step % self.config.checkpoint_every == 0:
                    self._save(f'step_{self.global_step}.pt', epoch)
                self.global_step += 1

            self._save(f'epoch_{epoch + 1}.pt', epoch + 1)
            logger.info(f'End of epoch {epoch}')



def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    # phonemizer/espeak logs a noisy 'words count mismatch' warning on every call.
    class _MutePhonemizer(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return 'words count mismatch' not in record.getMessage()

    for _handler in logging.getLogger().handlers:
        _handler.addFilter(_MutePhonemizer())

    args = Trainer.parse_args()
    config = Trainer.build_config(args)
    model_config = VitsModelConfig()
    trainer = Trainer(config, model_config, args)
    trainer.train()


if __name__ == '__main__':
    main()
