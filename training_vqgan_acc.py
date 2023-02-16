import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init, ImagePaths
from torch.utils.data import DataLoader
import GPUtil
# import wandb
from accelerate import Accelerator
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, list(range(torch.cuda.device_count())))))


class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device = args.device)
        self.discriminator = Discriminator(args).to(device = args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device = args.device)  #
        self.opt_vq, self.opt_disc = self.configure_optimizer(args)
        
        self.prepare_training()
        
        self.train(args)
        
    def configure_optimizer(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr = lr, eps = 1e-08, betas = (args.beta1, args.beta2)
        )    
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr = lr, eps = 1e-08, betas = (args.beta1, args.beta2))
        
        return opt_vq, opt_disc
    
    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok = True)
        os.makedirs("checkpoints", exist_ok = True)
        
    def train(self, args):
        train_data = ImagePaths(args.dataset_path, size = args.image_size)
        train_dataset = DataLoader(train_data, batch_size = args.batch_size, shuffle = False)
        accelerator = Accelerator()
        train_dataset, self.model, self.opt_vq_, self.opt_disc_ = accelerator.prepare(train_dataset, self.vqgan, self.opt_vq, self.opt_disc)



        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device = args.device)
                    
                    
                    # GPUtil.showUtilization()
                    decoded_images, _, q_loss = self.model(imgs)  # 여기서 out_of_memory

                    # GPUtil.showUtilization()  # 여기서 out_of_memory 
                    # print("#############################################")
                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.model.adopt_weight(args.disc_factor, epoch*steps_per_epoch + i, threshold = args.disc_start)
                    
                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)
                    
                    lambda_ = self.model.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor* lambda_ * g_loss
                    
                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)
                    
                    
                    self.opt_vq_.zero_grad()
                    accelerator.backward(vq_loss, retain_graph = True)
                    
                    
                    self.opt_disc_.zero_grad()
                    accelerator.backward(gan_loss)
                    
                    
                    self.opt_vq_.step()
                    self.opt_disc_.step()
                    
                    if i%10 == 0:
                        
                        with torch.no_grad():
                            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                            vutils.save_image(real_fake_images, os.path.join("results_acc", f"{epoch}_{i}.jpg"), nrow = 3)
                            
                    pbar.set_postfix(
                        VQ_Loss = np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss = np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                torch.save(self.model.state_dict(), os.path.join("checkpoints_acc",f"vqgan_epoch_{epoch}.pt"))
                
                
if __name__ == '__main__':
    accelerator = Accelerator()
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=512, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=2048, help='Number of codebook vectors (default: 1024)')  # number of codebook vector
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default= accelerator.device, help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=3, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    args = parser.parse_args()
    args.dataset_path = r"/workspace/VQGAN/data/dog"

        

    
    
    train_vqgan = TrainVQGAN(args)
