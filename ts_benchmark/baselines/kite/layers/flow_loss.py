import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .layers import KITEVectorField


class KITEFlowMatching(nn.Module):
    """
    Conditional Flow Matching with KITE's HCM, KGC, and CFG components.
    """
    def __init__(self, context_length, target_length, n_exo, n_head=8, depth=4, d_model=256, omega=1.0,
                 num_sampling_steps=50, p_uncond=0.1, rank=8, structure_max=0.7, min_sigma=0.1, fc_type="Linear",rate=2, use_future_exog=True, agg_method="mean", aux_loss_weight=0.1, mlp_ratio=4.0):
        super(KITEFlowMatching, self).__init__()

        self.in_channels = target_length
        self.net = KITEVectorField(
            context_length=context_length,
            n_exo = n_exo,
            model_channels=d_model,
            target_length=target_length,
            n_head=n_head,
            num_res_blocks=depth,
            rate=rate,
            use_future_exog=use_future_exog,
            mlp_ratio=mlp_ratio,
        )
        self.num_sampling_steps = num_sampling_steps
        self.p_uncond = p_uncond
        self.omega = omega
        self.rank = rank
        self.aux_loss_weight = aux_loss_weight
        
        self.manifold_basis = nn.Parameter(torch.empty(target_length, rank))
        nn.init.orthogonal_(self.manifold_basis)

        self.mix_param = nn.Parameter(torch.zeros(1))

        self.uncertainty_estimator = nn.Sequential(
            nn.LayerNorm(context_length),
            nn.Linear(context_length, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

        self.structure_max = structure_max
        self.agg_method = agg_method
        self.min_sigma = min_sigma
        nn.init.zeros_(self.uncertainty_estimator[-1].weight)
        nn.init.constant_(self.uncertainty_estimator[-1].bias, -2.5)

        if fc_type == 'Linear':
            self.barycenter_mapper = nn.Linear(context_length, target_length)
        elif fc_type == 'MLP':
            self.barycenter_mapper = nn.Sequential(nn.Linear(context_length, context_length*2), nn.ReLU(), nn.Linear(context_length*2, target_length))

    def get_hybrid_gaussian_noise(self, batch_size, device):
        z = torch.randn(batch_size, self.rank, device=device)
        raw_struct = torch.matmul(z, self.manifold_basis.T)
        struct_std = raw_struct.std(dim=1, keepdim=True) + 1e-6
        epsilon_struct = raw_struct / struct_std
        
        epsilon_iso = torch.randn(batch_size, self.in_channels, device=device)
        
        raw_alpha = torch.sigmoid(self.mix_param) * self.structure_max
        
        if self.training:
            mask = torch.bernoulli(torch.tensor(0.8, device=device))
            alpha = raw_alpha * mask
        else:
            alpha = raw_alpha
            
        epsilon_hybrid = alpha * epsilon_struct + (1 - alpha) * epsilon_iso
        
        final_std = epsilon_hybrid.std(dim=1, keepdim=True) + 1e-6
        epsilon_final = epsilon_hybrid / final_std
        
        return epsilon_final, alpha

    def forward(self, origin, target, x_exo, y_exo, prior, method1=False, method2=True):
        
        if self.training and random.random() < self.p_uncond:
            x_exo_in = torch.zeros_like(x_exo)
            y_exo_in = torch.zeros_like(y_exo)
            prior_in = torch.zeros_like(prior)
        else:
            x_exo_in = x_exo
            y_exo_in = y_exo
            prior_in = prior

        centroid = self.barycenter_mapper(origin)
        
        aux_loss = 0.0

        if method1:
            x0 = centroid + self.min_sigma * torch.randn_like(centroid)

        elif method2:
            batch_size = centroid.shape[0]
            
            epsilon, alpha = self.get_hybrid_gaussian_noise(batch_size, centroid.device)
            
            scale = F.softplus(self.uncertainty_estimator(origin)) + self.min_sigma
            
            x0 = centroid + epsilon * scale
            
            residuals = (target - centroid).detach()
            variance = scale.pow(2) + 1e-6
            
            nll = 0.5 * torch.log(variance) + 0.5 * residuals.pow(2) / variance
            
            aux_loss = nll.mean() * self.aux_loss_weight
            
        x1 = target
        t = torch.rand(centroid.shape[0], device=target.device)
        t_expand = t[:, None]
        
        xt = t_expand * x1 + (1 - t_expand) * x0
        v_target = x1 - x0
        
        v_pred = self.net(xt, t * 1000, x_exo_in, y_exo_in, prior_in)

        flow_loss = F.l1_loss(v_pred, v_target)

        return flow_loss + aux_loss
    
    @torch.no_grad()
    def sample(self, origin, x_exo, y_exo, prior, n_samples=10, method1=False, method2=True):
        centroid = self.barycenter_mapper(origin)
        B, D = centroid.shape
        
        centroid_exp = centroid.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, D)
        origin_exp = origin.repeat_interleave(n_samples, dim=0)
        
        x_exo_exp = x_exo.repeat_interleave(n_samples, dim=0)
        y_exo_exp = None if y_exo is None else y_exo.repeat_interleave(n_samples, dim=0)
        prior_exp = prior.repeat_interleave(n_samples, dim=0)

        if method1:
            x0 = centroid_exp + self.min_sigma * torch.randn_like(centroid_exp)
        
        elif method2:
            epsilon, _ = self.get_hybrid_gaussian_noise(B * n_samples, centroid.device)
            
            scale = F.softplus(self.uncertainty_estimator(origin_exp)) + self.min_sigma
            
            x0 = centroid_exp + epsilon * scale
        
        x = x0
        dt = 1.0 / self.num_sampling_steps
        
        null_x_exo = torch.zeros_like(x_exo_exp)
        null_y_exo = None if y_exo_exp is None else torch.zeros_like(y_exo_exp)
        null_prior = torch.zeros_like(prior_exp)

        for i in range(self.num_sampling_steps):
            t_val = i / self.num_sampling_steps
            t = torch.full((x.shape[0],), t_val, device=x.device)
            
            v_cond = self.net(x, t * 1000, x_exo_exp, y_exo_exp, prior_exp)
            v_uncond = self.net(x, t * 1000, null_x_exo, null_y_exo, null_prior)
            
            v_pred = v_uncond + self.omega * (v_cond - v_uncond)
            x = x + v_pred * dt
        
        x_reshaped = x.view(B, n_samples, D)
        if self.agg_method == "median":
            final_prediction = x_reshaped.median(dim=1).values
        else:
            final_prediction = x_reshaped.mean(dim=1)

        return final_prediction


FlowMatch = KITEFlowMatching
