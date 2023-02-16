import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN



class VQGANTransformer(nn.Module):
    def __init__(self, args):
        super(VQGANTransformer, self).__init__()
        
        self.sos_token = args.sos_token # start of sentence token
        
        self.vqgan = self.load_vqgan(args)
        
        transformer_config = {
            "vocab_size" : args.num_codebook_vectors,
            "block_size" : 1024,  # 몇으로 해야할꼬..??
            "n_layer" : 24,
            "n_head" : 16,
            "n_embd" : 1024
        }
        self.transformer = GPT(**transformer_config)
        
        self.pkeep = args.pkeep  # ratio of tokens that we keep. 
        
    # 정적 메소드 선언시 사용. 정적 메서드는 매개변수에 self를 지정하지 않는다. @은 데코레이션이라 하며 매서드(함수)가 추가 기능을 구현할 때 사용.
    # 메서드의 실행이 외부 상태에 영향을 끼치지 않는 순수 함수(pure function)를 만들 때 사용. 인스턴스의 상태를 변화시키지 않는 메서드를 만들 때 사용
    @staticmethod 
    def load_vqgan(args):
        model = VQGAN(args)
        model.load_checkpoint(args.checkpoint_path)
        model = model.eval()
        return model
    
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
        
    @torch.no_grad()
    def z_to_image(self, indices, p1 = 32, p2 = 32):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)  # p1, p2 dimension 맞춰야함(1, p1, p2, latent_dim)이라고 생각됨
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        # ix_to_vectors = ix_to_vectors.to("cuda")
        image = self.vqgan.decode(ix_to_vectors)
        return image.to('cpu')
        
    def forward(self, x):
        _, indices = self.encode_to_z(x)
        
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")  # 바꿔줘야할 수도 있음
        
        mask = torch.bernoulli(self.pkeep * torch.ones(indices.shape, device = indices.device))  # 50% 확률로 masking 하기
        mask = mask.round().to(dtype = torch.int64)
        random_indices = torch.randint_like(indices, self.transformer.config.vocab_size)
        new_indices = mask * indices + (1 - mask) * random_indices
        new_indices = new_indices.to("cuda")
        new_indices = torch.cat((sos_tokens, new_indices), dim = 1)
        
        target = indices
        
        logits, _ = self.transformer(new_indices[:,:-1])  # 마지막 열은 왜 안넣는거지..??
        
        return logits, target
    
    # pytorch에서 이미 제공하긴 하지만 이는 오직 k elements만을 보여줌. We will want to have a tensor which has a same dimensions as the input, and sets every elements to some really small number and keeps top k entries  
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf") 
        return out
    
    # transformer는 autoregressive이기에 (meaning we are generating each token one at a time)
    # sos token을 시작으로 모델이 다음에 어떤 토큰이 오는지를 예측함.
    # sos token과 first image token을 받음. -> end of with three tokens in total. 충분한 token을 받을 떄까지 계속 진행
    # normal image : 256, latent_dim : 16 x 16
    
    @torch.no_grad()
    def sample(self, x, c, steps, temperature= 1.0, top_k = 100):
        self.transformer.eval()
        c = c.to("cuda")
        x = torch.cat((c,x), dim = 1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            
            probs = F.softmax(logits, dim = -1)  # convert the predictions into probabilities using softmax
            
            ix = torch.multinomial(probs, num_samples = 1)
            
            x = torch.cat((x, ix), dim = 1)
            
        x = x[:,c.shape[1]:]  # cut off the sos token
        
        self.transformer.train()

        return x
    
    # vqgan can generate images much larger than we tried on 
    # why? : we don't generally have to stop after we have generated for example 256 tokens
    #        we could just go furher and generate 500 or even more and since our decoder is fully convolutional it doesn't care how many tokens it receives
    
    @torch.no_grad() 
    def log_images(self, x):  # Help us investigate the quality of the model 
        log = dict()
        
        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long()
        
        start_indices = indices[:, :indices.shape[1]//4]
        sample_indices = self.sample(start_indices, sos_tokens, steps = indices.shape[1] - start_indices.shape[1])
        quart_sample = torch.Tensor(self.z_to_image(sample_indices)).to('cuda')

        start_indices = indices[:, :indices.shape[1]//2]
        sample_indices = self.sample(start_indices, sos_tokens, steps = indices.shape[1] - start_indices.shape[1])
        half_sample = torch.Tensor(self.z_to_image(sample_indices)).to('cuda')
        
        start_indices = indices[:, :(indices.shape[1]//4)*3]
        sample_indices = self.sample(start_indices, sos_tokens, steps = indices.shape[1] - start_indices.shape[1])
        third_quart_sample = torch.Tensor(self.z_to_image(sample_indices)).to('cuda')
        
        start_indices = indices[:, :0]
        sample_indices = self.sample(start_indices, sos_tokens, steps = indices.shape[1])
        full_sample = torch.Tensor(self.z_to_image(sample_indices)).to('cuda')
        
        x_rec = torch.Tensor(self.z_to_image(indices)).to('cuda')

        log["input"] = x
        log["full_sample"] = full_sample
        log["quart_sample"] = quart_sample
        log["half_sample"] = half_sample
        log["third_quart_sample"] = third_quart_sample
        log["rec"] = x_rec
        
        return log, torch.concat((x, full_sample , quart_sample,half_sample, third_quart_sample, x_rec))  # outcome is dictionary which contains the normal image we've had in the reconstruction of this image which isn't using the transformer 
                                                                      # then the famous picutres where we cut off half of the image and let the transformer sample the bottom half and the a completely new sample image
    
    
