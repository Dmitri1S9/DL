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
from vits_finetune.losses import kl_loss, recon_loss
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
            )
            if (value := getattr(args, name, None)) is not None
        }
        return TrainingConfig(**overrides)


    @staticmethod
    def compute_loss(outputs: dict, config: TrainingConfig) -> tuple[torch.Tensor, dict]:
        """Combine recon + kl + duration into the total loss.
        """
        recon = recon_loss(outputs['predicted_mel'], outputs['target_mel'])
        kl = kl_loss(
            outputs['z_p'],
            outputs['posterior_log_stddev'],
            outputs['prior_mean'],
            outputs['prior_log_stddev'],
            outputs['z_mask'],
        )
        duration_loss = outputs['duration_loss']
        total_loss = config.mel_loss_weight * recon + config.kl_loss_weight * kl + duration_loss
        return total_loss, {
            'recon_loss': recon.item(),
            'kl_loss': kl.item(),
            'duration_loss': duration_loss.item(),
            'total_loss': total_loss.item(),
        }   
    
    def __init__(self, config: TrainingConfig, model_config: VitsModelConfig, args: argparse.Namespace):
            self.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
            logger.info(f'Using device: {self.device}')

            torch.manual_seed(config.seed)
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
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
            self.start_epoch, self.global_step = 0, 0
            self.config = config
            self.args = args


    def back_step_dec(func):
        def wrapper(self) -> None:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                for batch in self.dataloader:
                    self.optimizer.zero_grad()
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    func(self, batch) # IMPORTANT PART

                    if self.global_step % self.config.log_every == 0:
                        logger.info(f'Epoch {epoch} | Step {self.global_step} | ' + ' | '.join(f'{k}: {v:.4f}' for k, v in self.loss_dict.items()))
                    if self.global_step % self.config.checkpoint_every == 0:
                                    checkpoint_path = self.config.checkpoint_dir / f'step_{self.global_step}.pt'
                                    save_checkpoint(checkpoint_path, self.model, self.optimizer, self.global_step, epoch)
                                    logger.info(f'Saved checkpoint to {checkpoint_path}')
                    self.global_step += 1

                # Save checkpoint at end of epoch
                checkpoint_path = self.config.checkpoint_dir / f'epoch_{epoch + 1}.pt'
                save_checkpoint(checkpoint_path, self.model, self.optimizer, self.global_step, epoch + 1)
                logger.info(f'End of epoch {epoch} | Saved checkpoint to {checkpoint_path}')

        return wrapper
    

    @back_step_dec
    def stepof5GOAT(self, batch) -> None:
        # FORWARD
        outputs = self.model.forward_train(batch)
        # LOSS COMUPATION
        loss, self.loss_dict = self.compute_loss(outputs, self.config)
        # BACKWARD
        loss.backward()
        # STEP
        self.optimizer.step()


    def pretrain_check(func):
        def wrapper(self):
            if self.args.resume:
                ckpt = load_checkpoint(self.args.resume, self.model, self.optimizer, map_location=self.device)
                self.start_epoch = ckpt.get('epoch', 0)
                self.global_step = ckpt.get('step', 0)
                logger.info(f'Resumed from {self.args.resume} at epoch {self.start_epoch}, step {self.global_step}')
            return func(self)
        return wrapper


    @pretrain_check
    def train(self) -> None:
        """Run the fine-tuning loop end to end, including resume and checkpointing.
        """
        self.stepof5GOAT()



def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    args = Trainer.parse_args()
    config = Trainer.build_config(args)
    model_config = VitsModelConfig()
    trainer = Trainer(config, model_config, args)
    trainer.train()


if __name__ == '__main__':
    main()