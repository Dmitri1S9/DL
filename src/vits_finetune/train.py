from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from vits_finetune.checkpoint import load_checkpoint, save_checkpoint
from vits_finetune.collate import collate_fn
from vits_finetune.config import TrainingConfig, DiscriminatorConfig
from vits_finetune.dataset import VitsFinetuneDataset
from vits_finetune.losses import (kl_loss, recon_loss, discriminator_loss,
        feature_matching_loss, generator_adversarial_loss)
from vits_finetune.model import VitsFinetuneModel
from vits_finetune.model_config import VitsModelConfig
from vits_finetune.discriminator import Discriminator

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune VITS on a custom voice.')
    parser.add_argument('--dataset-repo-id', default=None)
    parser.add_argument('--checkpoint-dir', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--device', default=None)
    return parser.parse_args()

def build_config(args: argparse.Namespace) -> TrainingConfig:
    overrides = {
        name: value
        for name in ('dataset_repo_id', 'checkpoint_dir', 'batch_size',
                        'learning_rate', 'num_epochs')
        if (value := getattr(args, name, None)) is not None
    }
    return TrainingConfig(**overrides)

class Trainer:
    def __init__(self, config, model_config, disc_config, args):
        self.device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f'Using device: {self.device}')

        torch.manual_seed(config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model_name)
        self.dataset = VitsFinetuneDataset(config, self.tokenizer, split=config.train_split)
        self.dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=config.num_workers,
        )
        # Generator
        self.model_G = VitsFinetuneModel(model_config, config).to(self.device)
        self.optimizer_G = torch.optim.AdamW(self.model_G.parameters(), lr=config.learning_rate)
        # Discriminator
        self.discriminator = Discriminator(disc_config).to(self.device)
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=config.learning_rate)

        self.start_epoch, self.global_step = 0, 0
        self.config = config
        self.args = args

    def __pretrain_check(func):
            def wrapper(self):
                if self.args.resume:
                    ckpt = load_checkpoint(self.args.resume, self.model_G, self.optimizer_G,
                                        map_location=self.device)
                    d_resume = self.args.resume.parent / self.args.resume.name.replace('_G.pt', '_D.pt')
                    if d_resume.exists():
                        load_checkpoint(d_resume, self.discriminator, self.optimizer_D,
                                        map_location=self.device)
                        logger.info(f'Resumed discriminator from {d_resume.name}')
                    self.start_epoch = ckpt.get('epoch', 0)
                    self.global_step = ckpt.get('step', 0)
                    logger.info(f'Resumed from {self.args.resume} at epoch {self.start_epoch}, step {self.global_step}')
                return func(self)
            return wrapper

    def __back_step_dec(func):
        def wrapper(self) -> None:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                for batch in self.dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    d_loss, g_loss = func(self, batch)

                    self.loss_dict = {'d': d_loss.item(), 'g': g_loss.item()}

                    if self.global_step % self.config.log_every == 0:
                        logger.info(
                            f'Epoch {epoch} | Step {self.global_step} | '
                            + ' | '.join(f'{k}: {v:.4f}' for k, v in self.loss_dict.items())
                        )
                    if self.global_step % self.config.checkpoint_every == 0:
                        gp = self.config.checkpoint_dir / f'step_{self.global_step}_G.pt'
                        dp = self.config.checkpoint_dir / f'step_{self.global_step}_D.pt'
                        save_checkpoint(gp, self.model_G, self.optimizer_G, self.global_step, epoch)
                        save_checkpoint(dp, self.discriminator, self.optimizer_D, self.global_step, epoch)
                        logger.info(f'Saved checkpoints: {gp.name}, {dp.name}')
                    self.global_step += 1

                gp = self.config.checkpoint_dir / f'epoch_{epoch + 1}_G.pt'
                dp = self.config.checkpoint_dir / f'epoch_{epoch + 1}_D.pt'
                save_checkpoint(gp, self.model_G, self.optimizer_G, self.global_step, epoch + 1)
                save_checkpoint(dp, self.discriminator, self.optimizer_D, self.global_step, epoch + 1)
                logger.info(f'End of epoch {epoch} | Saved {gp.name}, {dp.name}')
        return wrapper
    
    def __log_parts(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            loss, parts = result
            if self.global_step % self.config.log_every == 0:
                logger.info('  parts | ' + ' | '.join(f'{k}: {v:.4f}' for k, v in parts.items()))
            return result
        return wrapper

    @__log_parts
    def __calculate_loss_g(self, outputs, fake_outs, real_fmaps, fake_fmaps) :
        dur = outputs['duration_loss']
        recon = recon_loss(outputs['predicted_mel'], outputs['target_mel'])
        kl = kl_loss(outputs['z_p'], outputs['posterior_log_stddev'],
                     outputs['prior_mean'], outputs['prior_log_stddev'], outputs['z_mask'])
        adv = generator_adversarial_loss(fake_outs)
        fm = feature_matching_loss(real_fmaps, fake_fmaps)
        g_loss = (self.config.mel_loss_weight * recon
                  + self.config.kl_loss_weight * kl + dur + adv + fm)
        parts = {'recon': recon.item(), 'kl': kl.item(), 'dur': dur.item(),
                 'adv': adv.item(), 'fm': fm.item()}
        return g_loss, parts  

    def __model_step(func):
        def wrapper(self, *args, **kwargs):
            optimizer, loss = func(self, *args, **kwargs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], 1.0
            )
            optimizer.step()
            return loss
        return wrapper

    @__model_step
    def __discriminator_step(self, real_wave, fake_wave):
        real_outs, _ = self.discriminator(real_wave)
        fake_outs, _ = self.discriminator(fake_wave.detach())
        d_loss = discriminator_loss(real_outs, fake_outs)
        return self.optimizer_D, d_loss 
    
    @__model_step
    def __generator_step(self, outputs, real_wave, fake_wave):
        fake_outs, fake_fmaps = self.discriminator(fake_wave)
        _, real_fmaps = self.discriminator(real_wave)
        g_loss, _ = self.__calculate_loss_g(outputs, fake_outs, real_fmaps, fake_fmaps)
        return self.optimizer_G, g_loss

    @__back_step_dec
    def stepof5GOATS(self, batch) -> None:
        self.model_G.train()
        self.discriminator.train()

        outputs = self.model_G.forward_train(batch)
        fake_wave = outputs['predicted_waveform']
        real_wave = outputs['target_waveform']

        # ===== Discriminator step =====
        d_loss = self.__discriminator_step(real_wave, fake_wave)

        # ===== Generator step =====
        g_loss = self.__generator_step(outputs, real_wave, fake_wave)

        return d_loss, g_loss

    @__pretrain_check
    def train(self) -> None:
        self.stepof5GOATS()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    args = parse_args()
    config = build_config(args)
    model_config = VitsModelConfig()
    trainer = Trainer(config, model_config, DiscriminatorConfig(), args)
    trainer.train()


if __name__ == '__main__':
    main()