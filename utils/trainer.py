"""
Training utilities for the CycleGAN-based harmonization project.

This module exposes `CycleGANTrainer`, a lightweight training loop
that coordinates two generators and two discriminators (A2B, B2A).
"""

import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from utils.utils import LambdaLR, ssim_batch, psnr_batch


class CycleGANTrainer:
    """
    Manage training and evaluation for CycleGAN models.

    Args:
        args: parsed CLI arguments (namespace) from `utils.arguments.get_args()`.
        args_list: grouped argument objects (used to dump argument values).
        train_dataset: dataset used for training iterations (Paired_Dataset).
        eval_dataset: dataset used for evaluation metrics and samples.
        generator1: generator model mapping vendor1 -> vendor2.
        discriminator1: discriminator for vendor1 images.
        generator2: generator mapping vendor2 -> vendor1.
        discriminator2: discriminator for vendor2 images.
        device: torch device to use for model training (`torch.device('cuda')`).
    """
    def __init__(self, args, args_list, train_dataset, eval_dataset,
                 generator1, discriminator1, generator2, discriminator2, device):
        # store config and device
        self.args = args
        self.device = device
        self.vendor1 = args.vendor1
        self.vendor2 = args.vendor2

        # prepare save directory structure for checkpoints, logs and samples
        self.save_dir = os.path.join(args.save_dir, args.experiment)
        for f in ['loss', 'model', 'sample']:
            os.makedirs(os.path.join(self.save_dir, f), exist_ok=True)

        # persist CLI arguments for reproducibility
        with open(os.path.join(self.save_dir, 'arguments.txt'), 'w') as f:
            for arg_group in args_list:
                f.write(arg_group.title + '\n')
                for action in arg_group._group_actions:
                    name, value = action.option_strings[-1], getattr(args, action.dest)
                    f.write(f'{name}: {value}\n')
                f.write('\n')

        # dataloaders: train uses shuffling and drop_last for stable batches
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
        # evaluation uses larger effective batch size and does not drop last
        self.eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size*4, shuffle=False, drop_last=False, num_workers=args.num_workers)

        # move models to device and keep references
        self.G_A2B = generator1.to(device)  # Vendor1 -> Vendor2 generator
        self.G_B2A = generator2.to(device)  # Vendor2 -> Vendor1 generator
        self.D_A = discriminator1.to(device)  # Vendor1 discriminator
        self.D_B = discriminator2.to(device)  # Vendor2 discriminator
        self.models = {
            'netG_{0}_to_{1}'.format(self.vendor1[0].upper(), self.vendor2[0].upper()): self.G_A2B,
            'netG_{0}_to_{1}'.format(self.vendor2[0].upper(), self.vendor1[0].upper()): self.G_B2A,
            'netD_{0}'.format(self.vendor1[0].upper()): self.D_A,
            'netD_{0}'.format(self.vendor2[0].upper()): self.D_B
        }

        # bookkeeping for losses & metrics tracking
        self.loss_types = ['Total', 'Adversarial', 'Cycle', 'Identity', 'Matching', 'Discriminator']
        self.metrics = ['SSIM A2B', 'PSNR A2B', 'SSIM B2A', 'PSNR B2A']
        columns = [f'Train {t} Loss' for t in self.loss_types] + [f'Eval {t} Loss' for t in self.loss_types] + [f'Eval {m}' for m in self.metrics]
        self.loss_df = pd.DataFrame(columns=columns)

        # loss functions used during training
        self.criterion_gan = nn.BCEWithLogitsLoss()  # stable GAN loss with logits
        self.criterion_l1 = nn.L1Loss()  # for reconstruction / identity / matching losses

        # optimizers and schedulers for generator and discriminators
        self.optim_G = torch.optim.Adam(itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optim_G, lr_lambda=LambdaLR(self.args.total_epoch, 0, self.args.decay_epoch).step)

        self.optim_D_A = torch.optim.Adam(self.D_A.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optim_D_A, lr_lambda=LambdaLR(self.args.total_epoch, 0, self.args.decay_epoch).step)

        self.optim_D_B = torch.optim.Adam(self.D_B.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
        self.scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optim_D_B, lr_lambda=LambdaLR(self.args.total_epoch, 0, self.args.decay_epoch).step)

    def train(self):
        """
        Run full training for the configured number of epochs.

        For each epoch this method runs a training pass, an evaluation pass,
        logs and saves losses/metrics, advances LR schedulers, and saves
        checkpoints according to `args.save_freq`.
        """
        for epoch in range(0, self.args.total_epoch):
            # training step
            epoch_losses = self.train_epoch(epoch)
            # evaluation step (adds evaluation losses/metrics)
            epoch_losses.update(self.eval_epoch(epoch))
            # persist losses and update plots
            self.update_losses(epoch, epoch_losses)

            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            # save checkpoints periodically
            if epoch % self.args.save_freq == 0 or epoch == self.args.total_epoch:
                self.save_model(epoch)
    
    def train_epoch(self, epoch):
        """
        Run a single training epoch and return aggregated losses.
        This is a thin wrapper around `epoch(..., train=True)`.
        """
        loss = self.epoch(epoch=epoch, data_loader=self.train_loader, train=True)
        return loss
    
    def eval_epoch(self, epoch):
        """
        Run a single evaluation epoch (no gradient updates).
        Returns the same loss/metric dictionary produced by `epoch`.
        """
        with torch.no_grad():
            loss = self.epoch(epoch=epoch, data_loader=self.eval_loader, train=False)
        return loss

    def epoch(self, epoch, data_loader, train=False):
        """
        Args:
            epoch (int): epoch index (for logging and saving).
            data_loader (DataLoader): the DataLoader to iterate over.
            train (bool): if True, perform optimizer updates; otherwise run in eval mode.

        Returns:
            dict: aggregated losses and metrics for this epoch.
        """
        mode = 'train' if train else 'eval'
        avg_tot_loss = 0
        avg_adv_loss = 0
        avg_ccl_loss = 0
        avg_identity_loss = 0
        avg_oml_loss = 0
        avg_disc_loss = 0
        avg_ssim_A2B, avg_ssim_B2A = 0, 0
        avg_psnr_A2B, avg_psnr_B2A = 0, 0

        if train:
            for model in self.models.values():
                model.train()
        else:
            for model in self.models.values():
                model.eval()
        
        data_iter = tqdm(enumerate(data_loader), desc='Epoch_{}:{}'.format(mode, epoch), total=len(data_loader), bar_format='{l_bar}{r_bar}')

        for i, data in data_iter:
            imgA, imgB = data[self.vendor1].to(self.device), data[self.vendor2].to(self.device)
            
            # Adversarial Loss
            # generate cross-domain images and compute discriminator logits
            imgA2B = self.G_A2B(imgA)
            imgB2A = self.G_B2A(imgB)
            class_B2A = self.D_A(imgB2A)
            class_A2B = self.D_B(imgA2B)
            # adversarial loss encourages generators to produce outputs
            # that discriminators classify as real (ones)
            loss_adv = self.criterion_gan(class_B2A, torch.ones_like(class_B2A).to(self.device)) + self.criterion_gan(class_A2B, torch.ones_like(class_B2A).to(self.device))

            # Cycle Consistency Loss: G_B2A(G_A2B(A)) ~ A and vice versa
            if self.args.ccl_weight > 0:
                imgA2B2A = self.G_B2A(imgA2B)
                imgB2A2B = self.G_A2B(imgB2A)
                loss_ccl = self.criterion_l1(imgA2B2A, imgA) + self.criterion_l1(imgB2A2B, imgB)
            else:
                loss_ccl = torch.tensor(0.0, device=self.device)

            # Identity Loss: encourage generator to behave like identity
            # when input already belongs to target domain
            if self.args.identity_weight > 0:
                imgA2A = self.G_B2A(imgA)
                imgB2B = self.G_A2B(imgB)
                loss_identity = self.criterion_l1(imgA2A, imgA) + self.criterion_l1(imgB2B, imgB)
            else:
                loss_identity = torch.tensor(0.0, device=self.device)

            # Original Matching Loss (OML): encourages direct matching between
            # translated output and the actual paired target image
            if self.args.oml_weight > 0:
                loss_oml = self.criterion_l1(imgA2B, imgB) + self.criterion_l1(imgB2A, imgA)
            else:
                loss_oml = torch.tensor(0.0, device=self.device)

            loss_G = loss_adv + loss_ccl*self.args.ccl_weight + loss_identity*self.args.identity_weight + loss_oml*self.args.oml_weight

            if train:
                self.optim_G.zero_grad()
                loss_G.backward()
                self.optim_G.step()
            
            pred_A_true = self.D_A(imgA)
            pred_A_false = self.D_A(self.G_B2A(imgB))
            loss_D_A = (self.criterion_gan(pred_A_true, torch.ones_like(pred_A_true).to(self.device)) + self.criterion_gan(pred_A_false, torch.zeros_like(pred_A_false).to(self.device))) / 2
            if train:
                self.optim_D_A.zero_grad()
                loss_D_A.backward()
                self.optim_D_A.step()
            
            pred_B_true = self.D_B(imgB)
            pred_B_false = self.D_B(self.G_A2B(imgA))
            loss_D_B = (self.criterion_gan(pred_B_true, torch.ones_like(pred_B_true).to(self.device)) + self.criterion_gan(pred_B_false, torch.zeros_like(pred_B_false).to(self.device))) / 2
            if train:
                self.optim_D_B.zero_grad()
                loss_D_B.backward()
                self.optim_D_B.step()
            
            avg_tot_loss += (loss_G.item() + loss_D_A.item() + loss_D_B.item())
            avg_adv_loss += loss_adv.item()
            avg_ccl_loss += loss_ccl.item()
            avg_identity_loss += loss_identity.item()
            avg_oml_loss += loss_oml.item()
            avg_disc_loss += (loss_D_A.item() + loss_D_B.item())

            if not train:
                avg_ssim_B2A += ssim_batch(imgB2A, imgA)
                avg_ssim_A2B += ssim_batch(imgA2B, imgB)
                avg_psnr_B2A += psnr_batch(imgB2A, imgA)
                avg_psnr_A2B += psnr_batch(imgA2B, imgB)

            post_fix = {"epoch": epoch,
                        "iter": i,
                        "AVG_Total_loss": avg_tot_loss / (i+1),
                        "AVG_Disc_loss": avg_disc_loss / (i+1),
                        "SSIM({}2{})".format(self.vendor2[0].upper(), self.vendor1[0].upper()): avg_ssim_B2A / (i+1),
                        "SSIM({}2{})".format(self.vendor1[0].upper(), self.vendor2[0].upper()): avg_ssim_A2B / (i+1),
                        "PSNR({}2{})".format(self.vendor2[0].upper(), self.vendor1[0].upper()): avg_psnr_B2A / (i+1),
                        "PSNR({}2{})".format(self.vendor1[0].upper(), self.vendor2[0].upper()): avg_psnr_A2B / (i+1),
                        "loss": loss_G.item() + loss_D_A.item() + loss_D_B.item()}
            
            if i % self.args.log_freq == 0:
                data_iter.write(str(post_fix))        
            
            # save example samples during evaluation at configured intervals
            if not train and i % self.args.sample_save_freq == 0:
                self.save_samples(epoch, i, imgA, imgB, imgA2B, imgB2A, imgA2B2A, imgB2A2B)
            
            if i == 30:
                break

        return {'{} Total Loss'.format(mode.capitalize()):avg_tot_loss / len(data_iter),
                '{} Adversarial Loss'.format(mode.capitalize()):avg_adv_loss / len(data_iter),
                '{} Cycle Loss'.format(mode.capitalize()):avg_ccl_loss / len(data_iter),
                '{} Identity Loss'.format(mode.capitalize()):avg_identity_loss / len(data_iter),
                '{} Matching Loss'.format(mode.capitalize()):avg_oml_loss / len(data_iter),
                '{} Discriminator Loss'.format(mode.capitalize()):avg_disc_loss / len(data_iter),
                '{} SSIM A2B'.format(mode.capitalize()):avg_ssim_A2B / (len(data_iter)),
                '{} SSIM B2A'.format(mode.capitalize()):avg_ssim_B2A / (len(data_iter)),
                '{} PSNR A2B'.format(mode.capitalize()):avg_psnr_A2B / (len(data_iter)),
                '{} PSNR B2A'.format(mode.capitalize()):avg_psnr_B2A / (len(data_iter))}

    def update_losses(self, epoch, loss_dict):
        """
        Persist epoch losses to CSV and update loss/metric plots.

        The method appends the provided `loss_dict` as a new row in
        `loss/loss.csv` and regenerates a PNG plot showing training vs
        evaluation losses and evaluation metrics.
        """
        print(loss_dict)
        self.loss_df.loc[epoch] = loss_dict
        self.loss_df.to_csv(os.path.join(self.save_dir, 'loss', 'loss.csv'))

        fig, axes = plt.subplots(1, len(self.loss_types) + len(self.metrics), figsize=(35, 6))

        for i, t in enumerate(self.loss_types):
            axes[i].plot(self.loss_df[f'Train {t} Loss'], label='train')
            axes[i].plot(self.loss_df[f'Eval {t} Loss'], label='eval')
            axes[i].legend()
            axes[i].set_title(f'{t} Loss')
        
        for j, m in enumerate(self.metrics):
            axes[len(self.loss_types) + j].plot(self.loss_df[f'Eval {m}'])
            axes[len(self.loss_types) + j].set_title(m)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss', 'loss.png'))
        plt.close()
    
    def save_model(self, epoch):
        """
        Save all models' state_dicts for the given epoch.

        Models are temporarily moved to CPU for checkpointing and then
        returned to the configured device to continue training.
        """
        for key, item in self.models.items():
            torch.save(item.cpu().state_dict(), os.path.join(self.save_dir, 'model', f'{key}_epoch_{epoch}.pth'))
            item.to(self.device)
        
    def save_samples(self, epoch, iter, imgA, imgB, imgA2B, imgB2A, imgA2B2A, imgB2A2B):
        """
        Save a grid of sample inputs and translated outputs for visual inspection.

        The function expects tensors in CHW format (batched) and will
        detach/move them to CPU before plotting. Note: the label for the
        original target domain uses `self.vendor1[2]` in the original
        code; this reproduces the prior behavior but may be a typo.
        """
        sample_save_dir = os.path.join(self.save_dir, 'sample', f'epoch_{epoch}')
        os.makedirs(sample_save_dir, exist_ok=True)

        imgs = [imgA, imgA2B, imgA2B2A, imgB, imgB2A, imgB2A2B]
        imgs = [img.detach().cpu().numpy() for img in imgs]
        labels = ['{0}'.format(self.vendor1[0].upper()),
                  '{0}2{1}'.format(self.vendor1[0].upper(), self.vendor2[0].upper()),
                  '{0}2{1}2{0}'.format(self.vendor1[0].upper(), self.vendor2[0].upper()),
                  '{0}'.format(self.vendor2[0].upper()),
                  '{0}2{1}'.format(self.vendor2[0].upper(), self.vendor1[0].upper()),
                  '{0}2{1}2{0}'.format(self.vendor2[0].upper(), self.vendor1[0].upper()),]

        batch_size = imgs[0].shape[0]
        fig, axes = plt.subplots(batch_size, 6, figsize=(12, 2*batch_size))

        for i, (img, label) in enumerate(zip(imgs, labels)):
            axes[0, i].set_title(label)
            for j in range(batch_size):
                axes[j, i].imshow(img[j, :, :, :].squeeze(), cmap='gray')
                axes[j, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_save_dir, f'iter_{iter}.png'))
        plt.close()
        