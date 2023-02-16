import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors) # Initializing Codebook Vector by Uniform Distribution
        
        
    def forward(self, z):  # Finding the minimal distance to the codebook vector
        z = z.permute(0, 2, 3, 1).contiguous()
        # z = z.to("cpu")
        z_flattened = z.view(-1, self.latent_dim)#.to("cpu")

        # Calculate the distances to the codebook vectors

        
        d = torch.sum(z_flattened**2, dim = 1, keepdim = True) + \
            torch.sum(self.embedding.weight**2, dim = 1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))
            
        min_encoding_indices = torch.argmin(d, dim = 1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # print(z_q.detach().device)
        # print(self.beta.device)
        # print(torch.mean((z_q.detach() - z) **2))
        # print(self.beta)
        # print(torch.mean((z_q - z.detach())**2))
        
        
        loss = torch.mean((z_q.detach() - z) **2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()
        
        z_q = z_q.permute(0, 3, 1, 2)
        # z_q = z_q.to("cuda")
        
        return z_q, min_encoding_indices, loss
        