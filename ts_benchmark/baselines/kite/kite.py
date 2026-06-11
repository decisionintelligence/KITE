from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from ts_benchmark.baselines.utils import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
)
from ts_benchmark.models.model_base import ModelBase
from ts_benchmark.baselines.utils import MLP, Conv, CrossAttention
from ts_benchmark.baselines.kite.models.KITEModel import KITEModel
from ts_benchmark.baselines.utils import (
    DBLoss,
)
from ..deep_forecasting_model_base import DeepForecastingModelBase

from ts_benchmark.baselines.kite.utils.tools import cal_prior_martix

MODEL_HYPER_PARAMS = {

    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "factor": 1,
    "patch_len": 16,
    "stride": 8,
    "activation": "gelu",
    "batch_size": 128,
    "lradj": "type3",
    "lr": 0.003,
    "norm": True,
    "num_epochs": 100,
    "num_workers": 0,
    "num_samples": 200,
    "num_sampling_steps": 10,
    "loss": "MAE",
    "dbloss_alpha": 0.2,
    "dbloss_beta": 0.5,
    "aux_loss_weight": 0.1,
    "patience": 8,
    "alpha": 0.2,
    "beta": 0.1,
    "use_c_exog": True,
    "use_t_exog": True,
    "use_c": True,
    "use_t": True,
    "warmup_epoch":10,
    "infer_use_future": True,
    "structure_max":0.9,
    "rank":8,
    "fc_type":"Linear",
    "min_sigma":0.15,
    "piror": "pearson",
    "prior_level": "sample",
    'rate':12,
    'use_future_exog':True,
    'agg_method':'mean',
    'unknown_future_exog_strategy':'zero',
    'dropout': 0.0,
    'flow_depth': 2,
    'flow_dim': 256,
    'flow_head': 4,
    'noise_dropout': 0.1,
    'omega': 1.0,
    'p_uncond': 0.1,
    'mlp_ratio': 4.0,
}


class KITE(DeepForecastingModelBase):
    """
    Benchmark adapter for KITE.
    """

    def __init__(self, **kwargs):
        super(KITE, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "KITE"

    def _init_criterion(self):
        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        elif self.config.loss == "DBLoss":
            criterion = DBLoss(self.config.dbloss_alpha, self.config.dbloss_beta)
        else:
            criterion = nn.HuberLoss(delta=0.5)
        self.config.criterion = criterion
        return criterion

    def _init_model(self):
        return KITEModel(self.config)

    def validate(
        self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data‘s dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        total_loss = []
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                x_exo = input[:, :, series_dim:].to(device)
                x_endo = input[:, :, :series_dim].to(device)
                y_exo = target[:, -config.horizon :, series_dim:].to(device)
                y_endo = target[:, -config.horizon :, :series_dim].to(device)

                out_loss = self.model.train_function(x_endo = x_endo, x_exo=x_exo, y_endo=y_endo, y_exo = y_exo)
                out_loss = out_loss.detach().cpu()
                total_loss.append(out_loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        if self.CovariateFusion is not None:
            self.CovariateFusion.train()
        return total_loss
    
    def _init_optimizer(self,):
        return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        **kwargs,
    ) -> "ModelBase":
        """
        Train the model.
        :param train_valid_data: Time series data used for training and validation.
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        if covariates is None:
            covariates = {}
        series_dim = train_valid_data.shape[-1]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            train_valid_data = pd.concat([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-1]
        else:
            exog_dim = 0

        if train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim

        criterion = self._init_criterion()
        self.model = self._init_model()
        if self.config.fusion_method == "mlp":
            self.CovariateFusion = MLP(self.config)
        elif self.config.fusion_method == "cross_attention":
            self.CovariateFusion = CrossAttention(self.config)
        elif self.config.fusion_method == "conv":
            self.CovariateFusion = Conv(self.config)
        else:
            self.CovariateFusion = None
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            if self.CovariateFusion is not None:
                self.CovariateFusion = nn.DataParallel(
                    self.CovariateFusion, device_ids=device_ids
                )
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )

        if exog_dim > 0:
            self.scaler1.fit(train_data.values[:, :series_dim])
            self.scaler2.fit(train_data.values[:, series_dim:])

            if config.norm:
                scaled_series = self.scaler1.transform(
                    train_data.values[:, :series_dim]
                )
                scaled_exog = self.scaler2.transform(train_data.values[:, series_dim:])
                final_train_data = np.concatenate((scaled_series, scaled_exog), axis=1)
                train_data = pd.DataFrame(
                    final_train_data,
                    columns=train_data.columns,
                    index=train_data.index,
                )
        else:
            self.scaler1.fit(train_data.values)
            if config.norm:
                train_data = pd.DataFrame(
                    self.scaler1.transform(train_data.values),
                    columns=train_data.columns,
                    index=train_data.index,
                )

        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    scaled_series = self.scaler1.transform(
                        valid_data.values[:, :series_dim]
                    )
                    scaled_exog = self.scaler2.transform(
                        valid_data.values[:, series_dim:]
                    )
                    final_valid_data = np.concatenate(
                        (scaled_series, scaled_exog), axis=1
                    )
                    valid_data = pd.DataFrame(
                        final_valid_data,
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                else:
                    valid_data = pd.DataFrame(
                        self.scaler1.transform(valid_data.values),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
            )

        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(config, "prior_level", "sample") == "dataset":
            train_data_tensor = torch.tensor(train_data.values)
            prior = cal_prior_martix(
                train_data_tensor[:, -1].unsqueeze(0).unsqueeze(-1),
                train_data_tensor[:, :-1].unsqueeze(0),
                method=config.piror,
            )
            self.model.dataset_prior = prior.to(device).float()
        else:
            self.model.dataset_prior = None
        optimizer = self._init_optimizer()

        if config.use_amp == 1:
            scaler = torch.cuda.amp.GradScaler()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if self.CovariateFusion is not None:
            total_params += sum(
                p.numel() for p in self.CovariateFusion.parameters() if p.requires_grad
            )
        print(f"Total trainable parameters: {total_params}")

        if 1==1:
            for epoch in range(config.num_epochs):
                self.epoch = epoch
                self.model.train()
                if self.CovariateFusion is not None:
                    self.CovariateFusion.train()
                for i, (input, target, input_mark, target_mark) in enumerate(
                    self.train_data_loader
                ):
                    optimizer.zero_grad()
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )
                    x_exo = input[:, :, series_dim:].to(device)
                    x_endo = input[:, :, :series_dim].to(device)
                    y_exo = target[:, -config.horizon :, series_dim:].to(device)
                    y_endo = target[:, -config.horizon :, :series_dim].to(device)

                    out_loss = self.model.train_function(x_endo = x_endo, x_exo=x_exo, y_endo=y_endo, y_exo = y_exo)


                    total_loss = out_loss

                    if config.use_amp == 1:
                        scaler.scale(total_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    if self.config.lradj == "TST":
                        self._adjust_lr(optimizer, epoch + 1, config)

                if train_ratio_in_tv != 1:
                    valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                    improved = self.early_stopping(valid_loss, self.model)
                    if improved:
                        if self.CovariateFusion is not None:
                            self.check_point = self.save_checkpoint(
                                {
                                    "Model": self.model,
                                    "CovariateFusion": self.CovariateFusion,
                                }
                            )
                        else:
                            self.check_point = self.save_checkpoint({"Model": self.model})
                    if self.early_stopping.early_stop:
                        break

                if self.config.lradj != "TST":
                        self._adjust_lr(optimizer, epoch + 1, config)
                        

    def _perform_rolling_predictions(
            self,
            horizon: int,
            input_np: np.ndarray,
            exog_future: torch.Tensor,
            series_dim: int,
            all_mark: np.ndarray,
            device: torch.device,
        ) -> list:
            """
            Perform rolling predictions using the given input data and marks.

            :param horizon: Length of predictions to be made.
            :param input_np: Numpy array of input data.
            :param exog_future: Future exogenous data used for prediction.
            :param series_dim: Dimension of the series data.
            :param all_mark: Numpy array of all marks (time stamps mark).
            :param device: Device to run the model on.
            :return: List of predicted results for each prediction batch.
            """
            rolling_time = 0
            input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
                input_np, None, all_mark, rolling_time
            )
            if not getattr(self.config, "use_future_exog", True):
                exog_future = None
            if exog_future is not None:
                rolling_time_sum = horizon // self.config.horizon + 1
                need_horizon = rolling_time_sum * self.config.horizon - horizon
                exog_future = torch.cat(
                    (
                        exog_future,
                        torch.zeros(
                            (exog_future.shape[0], need_horizon, exog_future.shape[-1])
                        ).to(device),
                    ),
                    dim=1,
                )
                exog_future = exog_future.float()
            with torch.no_grad():
                answers = []
                while not answers or sum(a.shape[1] for a in answers) < horizon:
                    input, dec_input, input_mark, target_mark = (
                        torch.tensor(input_np, dtype=torch.float32).to(device),
                        torch.tensor(target_np, dtype=torch.float32).to(device),
                        torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                        torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                    )
                    if exog_future is not None:
                        exog_future1 = exog_future[
                            :,
                            rolling_time
                            * self.config.horizon : (rolling_time + 1)
                            * self.config.horizon,
                            :,
                        ]
                    else:
                        exog_future1 = None
                    x_exo = input[:, :, series_dim:]
                    x_endo = input[:, :, :series_dim]

                    y_exo = exog_future1
                    output = self.model.inference(x_endo = x_endo, x_exo=x_exo,  y_exo = y_exo, num_samples = self.config.num_samples)

                    output1 = output[:, -self.config.horizon :, :series_dim]
                    real_batch_size = output.shape[0]
                    exog_dim = input.shape[-1] - series_dim
                    if exog_future1 is not None:
                        future_exog_for_roll = exog_future1
                    elif exog_dim > 0:
                        fill_strategy = getattr(self.config, "unknown_future_exog_strategy", "zero")
                        if fill_strategy == "last":
                            future_exog_for_roll = x_exo[:, -1:, :].expand(-1, self.config.horizon, -1)
                        else:
                            future_exog_for_roll = torch.zeros(
                                real_batch_size,
                                self.config.horizon,
                                exog_dim,
                                device=output.device,
                                dtype=output.dtype,
                            )
                    else:
                        future_exog_for_roll = output[:, -self.config.horizon :, series_dim:]
                    output = torch.cat([output1, future_exog_for_roll], dim=-1)
                    column_num = output.shape[-1]
                    answer = (
                        output1.cpu()
                        .numpy()
                        .reshape(real_batch_size, -1, series_dim)[
                            :, -self.config.horizon :, :
                        ]
                    )
                    answers.append(answer)
                    if sum(a.shape[1] for a in answers) >= horizon:
                        break
                    rolling_time += 1
                    output = output.cpu().numpy()[:, -self.config.horizon :, :]
                    (
                        input_np,
                        target_np,
                        input_mark_np,
                        target_mark_np,
                    ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

            answers = np.concatenate(answers, axis=1)
            return answers[:, -horizon:, :]


ForecastingFlow = KITE
