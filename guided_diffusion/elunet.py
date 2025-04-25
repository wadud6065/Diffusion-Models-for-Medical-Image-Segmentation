import math
import torch
import torch.nn as nn
from .elunet_parts import DoubleConv,DownSample,UpSample,OutConv

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ELUnet(nn.Module):
    def __init__(self,in_channels,out_channels,n:int = 8) -> None:
        """ 
        Construct the Elu-net model.
        Args:
            in_channels: The number of color channels of the input image. 0:for binary 3: for RGB
            out_channels: The number of color channels of the input mask, corresponds to the number
                            of classes.Includes the background
            n: Channels size of the first CNN in the encoder layer. The bigger this value the bigger 
                the number of parameters of the model. Defaults to n = 8, which is recommended by the 
                authors of the paper.
        """
        super().__init__()

        self.const = n
        time_embed_dim = 16*n

        # ------ Input convolution --------------
        self.in_conv = DoubleConv(in_channels,n)
        # -------- Encoder ----------------------
        self.down_1 = DownSample(n, 2*n, time_embed_dim)
        self.down_2 = DownSample(2*n, 4*n, time_embed_dim)
        self.down_3 = DownSample(4*n, 8*n, time_embed_dim)
        self.down_4 = DownSample(8*n, 16*n, time_embed_dim)
        
        # -------- Upsampling ------------------
        self.up_1024_512 = UpSample(16*n, 8*n, 2, time_embed_dim)

        self.up_512_64 = UpSample(8*n, n, 8, time_embed_dim)
        self.up_512_128 = UpSample(8*n, 2*n, 4, time_embed_dim)
        self.up_512_256 = UpSample(8*n, 4*n, 2, time_embed_dim)
        self.up_512_512 = UpSample(8*n, 8*n, 0, time_embed_dim)

        self.up_256_64 = UpSample(4*n, n, 4, time_embed_dim)
        self.up_256_128 = UpSample(4*n, 2*n, 2, time_embed_dim)
        self.up_256_256 = UpSample(4*n, 4*n, 0, time_embed_dim)

        self.up_128_64 = UpSample(2*n, n, 2, time_embed_dim)
        self.up_128_128 = UpSample(2*n, 2*n, 0, time_embed_dim)

        self.up_64_64 = UpSample(n, n, 0, time_embed_dim)
     
        # ------ Decoder block ---------------
        self.dec_4 = DoubleConv(2*8*n,8*n)
        self.dec_3 = DoubleConv(3*4*n,4*n)
        self.dec_2 = DoubleConv(4*2*n,2*n)
        self.dec_1 = DoubleConv(5*n,n)
        # ------ Output convolution

        self.out_conv = OutConv(n,out_channels)

        self.time_embed = nn.Sequential(
            nn.Linear(n, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x, time):

        x = self.in_conv(x) # 64
        emb = self.time_embed(timestep_embedding(time, self.const))

        # ---- Encoder outputs
        x_enc_1 = self.down_1(x, emb) # 128
        x_enc_2 = self.down_2(x_enc_1, emb) # 256
        x_enc_3 = self.down_3(x_enc_2, emb) # 512
        x_enc_4 = self.down_4(x_enc_3, emb) # 1024
    
        # ------ decoder outputs
        x_up_1 = self.up_1024_512(x_enc_4, emb)
        x_dec_4 = self.dec_4(torch.cat([x_up_1,self.up_512_512(x_enc_3, emb)],dim=1))

        x_up_2 = self.up_512_256(x_dec_4, emb)
        x_dec_3 = self.dec_3(torch.cat([x_up_2,
            self.up_512_256(x_enc_3, emb),
            self.up_256_256(x_enc_2, emb)
            ],
        dim=1))

        x_up_3 = self.up_256_128(x_dec_3, emb)
        x_dec_2 = self.dec_2(torch.cat([
            x_up_3,
            self.up_512_128(x_enc_3, emb),
            self.up_256_128(x_enc_2, emb),
            self.up_128_128(x_enc_1, emb)
        ],dim=1))

        x_up_4 = self.up_128_64(x_dec_2, emb)
        x_dec_1 = self.dec_1(torch.cat([
            x_up_4,
            self.up_512_64(x_enc_3, emb),
            self.up_256_64(x_enc_2, emb),
            self.up_128_64(x_enc_1, emb),
            self.up_64_64(x, emb)
        ],dim=1))

        return self.out_conv(x_dec_1)