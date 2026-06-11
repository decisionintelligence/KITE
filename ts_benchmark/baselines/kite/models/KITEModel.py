import torch
from torch import nn
from ..layers.flow_loss import KITEFlowMatching

from einops import rearrange, repeat


class KITEModel(nn.Module):
    def __init__(self, configs, **kwargs):

        super().__init__()

        context_window = configs.seq_len
        target_window = configs.horizon
        d_model = configs.flow_dim
        num_sampling_steps = configs.num_sampling_steps
        depth = configs.flow_depth
        flow_head = configs.flow_head
        self.omega = configs.omega
        self.noise_dropout = nn.Dropout(configs.noise_dropout)
        self.flow_matching =  KITEFlowMatching(context_window, target_window, n_exo=configs.input_dim-configs.output_dim, n_head=flow_head, depth=depth, d_model=d_model,num_sampling_steps=num_sampling_steps, omega=configs.omega, p_uncond=configs.p_uncond,structure_max=configs.structure_max, rank=configs.rank, min_sigma=configs.min_sigma, fc_type=configs.fc_type, rate=configs.rate,
                                        use_future_exog = configs.use_future_exog, agg_method=getattr(configs, 'agg_method', 'mean'), aux_loss_weight=getattr(configs, 'aux_loss_weight', 0.1), mlp_ratio=getattr(configs, 'mlp_ratio', 4.0))

        self.prior_level = getattr(configs, 'prior_level', 'sample')
        self.dataset_prior = None

    def _resolve_prior(self, x, x_exo):
        if self.prior_level == 'dataset':
            assert self.dataset_prior is not None, \
                "prior_level='dataset' requires dataset_prior to be initialized in forecast_fit"
            B, _, C_in = x.shape
            return self.dataset_prior.expand(B * C_in, -1, -1).to(x.device, x.dtype)
        return cal_pearson_martix(x, x_exo)


    def train_function(self, x_endo, x_exo, y_endo, y_exo=None):
        x = x_endo
        y = y_endo
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)
        y = (y - seq_mean) / torch.sqrt(seq_var)

        exo_mean = torch.mean(x_exo, dim=1, keepdim=True)
        exo_var = torch.var(x_exo, dim=1, keepdim=True) + 1e-5
        x_exo = (x_exo - exo_mean) / torch.sqrt(exo_var)
        if y_exo is not None:
            y_exo = (y_exo - exo_mean) / torch.sqrt(exo_var)
        
        prior = self._resolve_prior(x, x_exo)

        x = rearrange(x,'b t c -> (b c) t')
        y = rearrange(y,'b t c -> (b c) t')

        x_exo = rearrange(x_exo,'b t c -> b c t')
        if y_exo is not None:
            y_exo = rearrange(y_exo,'b t c -> b c t')

        h_x = x
        h_y = y

        self.noise_dropout(h_x)
        loss = self.flow_matching(h_x, h_y, x_exo, y_exo, prior)
        return loss
    
    @torch.no_grad()
    def inference(self, x_endo, x_exo, y_exo=None, num_samples=50):
        use_future_exog = self.flow_matching.net.use_future_exog
        if use_future_exog and y_exo is None:
            raise ValueError("y_exo is required when use_future_exog=True")

        x = x_endo
        c = x.size(-1)
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = (x - seq_mean) / torch.sqrt(seq_var)
        exo_mean = torch.mean(x_exo, dim=1, keepdim=True)
        exo_var = torch.var(x_exo, dim=1, keepdim=True) + 1e-5
        x_exo = (x_exo - exo_mean) / torch.sqrt(exo_var)
        if y_exo is not None:
            y_exo = (y_exo - exo_mean) / torch.sqrt(exo_var)

        prior = self._resolve_prior(x, x_exo)

        x_exo = rearrange(x_exo, 'b t c -> b c t')
        if y_exo is not None:
            y_exo = rearrange(y_exo,'b t c -> b c t')
        x = rearrange(x,'b t c -> (b c) t')
        h_x = x

        y = self.flow_matching.sample(h_x, x_exo, y_exo, prior,n_samples = num_samples)
        
        y = rearrange(y,'(b c) t -> b t c',c=c)
        y = y * torch.sqrt(seq_var) + seq_mean
        return y

def cal_pearson_martix(x, x_exo):
    x_endo_for_corr = rearrange(x, 'b t c -> (b c) 1 t')
    B, T, C_in = x.shape
    _, _, C_exo = x_exo.shape
    
    x_exo_for_corr = rearrange(x_exo, 'b t c -> b c t') # [B, C_exo, T]
    x_exo_for_corr = repeat(x_exo_for_corr, 'b c_exo t -> (b c_in) c_exo t', c_in=C_in)

    A = x_endo_for_corr
    B = x_exo_for_corr
    A_c = A - A.mean(dim=-1, keepdim=True)
    B_c = B - B.mean(dim=-1, keepdim=True)
    
    covariance = torch.matmul(A_c, B_c.transpose(-2, -1))  # [B, 1, Lc]
    
    A_std = torch.sqrt((A_c ** 2).sum(dim=-1, keepdim=True))
    B_std = torch.sqrt((B_c ** 2).sum(dim=-1)).unsqueeze(1)
    
    eps = 1e-8
    return (covariance / (A_std * B_std + eps)).abs()

def cal_granger_matrix(x, x_exo, lag=4):
    """
    Estimate Granger-style exogenous knowledge with linear regression weights.
    """
    B, T, C_in = x.shape
    _, _, C_exo = x_exo.shape
    
    x_endo_reg = rearrange(x, 'b t c -> (b c) t 1')
    
    x_exo_reg = repeat(x_exo, 'b t c -> (b c_in) t c', c_in=C_in)

    y = x_endo_reg[:, lag:, :]
    feat_endo = x_endo_reg[:, :-lag, :]
    feat_exo = x_exo_reg[:, :-lag, :]
    A = torch.cat([feat_endo, feat_exo], dim=-1)
    
    A_T = A.transpose(1, 2)
    ATA = torch.bmm(A_T, A)
    
    reg_lambda = 1e-4
    identity = torch.eye(ATA.shape[1], device=x.device).unsqueeze(0)
    ATA = ATA + reg_lambda * identity
    
    ATy = torch.bmm(A_T, y)
    
    try:
        w = torch.linalg.solve(ATA, ATy)
    except:
        w = torch.matmul(torch.linalg.pinv(ATA), ATy)
        
    w_exo = w[:, 1:, 0]
    
    return w_exo.unsqueeze(1).abs()


Model = KITEModel
