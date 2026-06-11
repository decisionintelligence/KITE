import torch
import torch.nn as nn
import math


def modulate(x, shift, scale):
    if len(x.shape)>len(scale.shape):
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ResBlock(nn.Module):
    """
    Residual MLP block with adaptive layer normalization.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    Final projection layer for the KITE vector field.
    """

    def __init__(self, model_channels, out_channels, n_exo, rate=2,use_future_exog=True):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        
        self.fc2 = nn.Linear(model_channels, out_channels, bias=True)
        self.use_future_exog = use_future_exog
        if use_future_exog:
            self.fc1 = nn.Sequential(nn.Linear(n_exo + 1, rate * (n_exo + 1)), nn.ReLU(), nn.Linear(rate*(n_exo + 1),1))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

 
    def forward(self, x, c):
        x = x.permute(0, 2, 1)
        if self.use_future_exog:
            x = self.fc1(x).squeeze(-1)
        else:
            x = x[..., 0]
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.fc2(x)
        return x

class SiTBlock(nn.Module):
    """
    KGC block with adaptive normalization and knowledge-guided attention.
    """
    def __init__(self, hidden_size, num_heads, n_exo, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = KnowledgeGuidedAttention(hidden_size,num_heads=num_heads, n_exo=n_exo) 
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim), nn.GELU(approximate="tanh") ,nn.Linear(mlp_hidden_dim, hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, exo, prior):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
       
        x = x + gate_msa * self.attn(x_q=modulate(self.norm1(x), shift_msa, scale_msa).unsqueeze(1), x_kv=exo, w_ij=prior).squeeze(1)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class KITEVectorField(nn.Module):
    """
    Vector field network used by KITE Conditional Flow Matching.
    """

    def __init__(
        self,
        context_length,
        n_exo,
        model_channels,
        target_length,
        n_head,
        num_res_blocks,
        rate=2,
        use_future_exog=True,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.in_channels = target_length
        self.model_channels = model_channels
        self.out_channels = target_length
        self.num_res_blocks = num_res_blocks

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed_x = nn.ModuleList([SiTBlock(model_channels,num_heads=n_head, n_exo=n_exo, mlp_ratio=mlp_ratio) for _ in range(num_res_blocks)])

        if use_future_exog:
            self.cond_embed_y = nn.ModuleList([SiTBlock(model_channels,num_heads=n_head, n_exo=n_exo, mlp_ratio=mlp_ratio) for _ in range(num_res_blocks)])
            self.y_exo_embedding = nn.Linear(target_length, model_channels)

        self.input_proj = nn.Linear(target_length, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels
            )) 

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, target_length, n_exo,rate=rate,use_future_exog=use_future_exog)
        self.use_future_exog = use_future_exog
        self.x_exo_embedding = nn.Linear(context_length, model_channels,)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.fc2.weight, 0)
        nn.init.constant_(self.final_layer.fc2.bias, 0)


    def forward(self, x0, t, x_exo, y_exo, prior):
        """
        Predict the velocity field for the interpolated state.
        """

        x = self.input_proj(x0)
        t = self.time_embed(t)
        if self.use_future_exog:
            y_exo = self.y_exo_embedding(y_exo)
        x_exo = self.x_exo_embedding(x_exo)

        prior = torch.softmax(prior, dim=-1)
        for index, block in enumerate(self.res_blocks):
            x = self.cond_embed_x[index](x, t, x_exo, prior)
            if self.use_future_exog:
                x = self.cond_embed_y[index](x, t, y_exo, prior)
        if self.use_future_exog:
            x = torch.cat([x.unsqueeze(1), y_exo], dim=1)
        else:
            x = x.unsqueeze(1)
        return self.final_layer(x, t)
    
class KnowledgeGuidedAttention(nn.Module):
    """
    Knowledge-Guided Attention with bilinear statistical-prior modulation.
    """
    def __init__(self, dim, n_exo, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.pos_emb = nn.Parameter(torch.randn(1, n_exo, dim))

        self.w1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.w2 = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.proj_drop = nn.Dropout(proj_drop)
        self.prior_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_q, x_kv, w_ij):
        """
        x_q: [B, N_q, C], endogenous query states.
        x_kv: [B, N_k, C], exogenous key/value states.
        w_ij: [B, 1, N_k], statistical knowledge prior.
        """
        B, N_q, C = x_q.shape
        _, N_k, _ = x_kv.shape
        head_dim = C // self.num_heads

        x_kv = x_kv + self.pos_emb[:, :, :]
        k = self.wk(x_kv).reshape(B, N_k, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.wv(x_kv).reshape(B, N_k, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q1 = self.w1(x_q).reshape(B, N_q, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q2 = self.w2(x_q).reshape(B, N_q, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn_base = (q1 @ k.transpose(-2, -1)) * self.scale
        
        attn_prior_learnable = (q2 @ k.transpose(-2, -1)) * self.scale
        
        w_ij_expanded = w_ij.unsqueeze(1) if w_ij.dim() == 3 else w_ij
        
        w_ij_log = torch.log(w_ij_expanded + 1e-6) 
        
        attn = attn_base + (attn_prior_learnable * w_ij_expanded) + w_ij_log
                
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj_drop(self.proj(x))
        return x


SimpleMLPAdaLN = KITEVectorField
WeightedGraphAttention = KnowledgeGuidedAttention
