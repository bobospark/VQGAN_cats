import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
import GPUtil

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device = args.device)
        self.decoder = Decoder(args).to(device = args.device)
        self.codebook = Codebook(args).to(device = args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device = args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device = args.device)
        
        
        
        
    def forward(self, imgs):
        
        encoded_images = self.encoder(imgs)
        # print("#######", encoded_images.size())  # [1, 256, 32, 32]  # [#, latent_dim, encoder_output_size, encoder_output_size]

        quant_conv_encoded_images = self.quant_conv(encoded_images)
        # print('########', quant_conv_encoded_images.size())  # [1, 256, 32, 32]
        
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        # print("#######",codebook_mapping.size())  # [1, 256, 32, 32]
        
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        # print(post_quant_conv_mapping.size())  # [1, 256, 32, 32]
        
        decoded_images = self.decoder(post_quant_conv_mapping)  # 여기서 out_of_memory Decoder가 decoded_image를 512x512로 나타내야하는거 아닌가?
        # print(decoded_images.size())  # size = [1, image_channels, image_size, image_size]
        
        return decoded_images, codebook_indices, q_loss
    
    
    def encode(self,imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss
    

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images
    # GPUtil
    # Calculate the lambda
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph = True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
        
        lamda_ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lamda_ = torch.clamp(lamda_, 0, 1e4).detach()
        
        return 0.8 * lamda_
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value = 0. ):
        if i < threshold:
            disc_factor = value
        return disc_factor
    
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))