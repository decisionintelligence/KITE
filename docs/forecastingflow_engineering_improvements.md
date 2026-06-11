# ForecastingFlow 工程改进备忘

仅工程层面的改进，不改动论文方法。按"风险 / 收益比"从高到低排列，前几条强烈推荐，后几条按需取。

涉及文件：
- `ts_benchmark/baselines/forecastingflow/models/ForecastingFlowModel.py`
- `ts_benchmark/baselines/forecastingflow/layers/flow_loss.py`
- `ts_benchmark/baselines/forecastingflow/forecastingflow.py`

---

## 1. 🔴 noise_dropout 是 bug，根本没生效

**文件**：`ForecastingFlowModel.py:68`（`train_function` 内）

**现状**：
```python
self.noise_dropout(h_x)
```
`nn.Dropout` 不是 in-place，返回值被丢了。设了 `noise_dropout > 0` 也等于没设。

**改法**：
```python
h_x = self.noise_dropout(h_x)
```

**预期**：如果配置里 `noise_dropout` 不为 0，激活后通常涨点。论文若提了"对回看窗口噪声正则"，这就是把它真正打开。

---

## 2. 🔴 Structure Dropout 退化成 batch-level，应改 sample-level

**文件**：`flow_loss.py:83-86`（`get_hybrid_gaussian_noise`）

**现状**：
```python
if self.training:
    mask = torch.bernoulli(torch.tensor(0.8, device=device))   # 标量
    alpha = raw_alpha * mask
```
全 batch 共用同一个 0/1 —— 要么都丢结构化噪声，要么都不丢。粒度过粗。

**改法**：
```python
if self.training:
    mask = torch.bernoulli(torch.full((batch_size, 1), 0.8, device=device))
    alpha = raw_alpha * mask
```

**预期**：论文里只说"以 20% 概率丢弃结构"，未规定粒度。改到 sample-level 更符合常规理解，正则更细，通常稳涨。

---

## 3. 🟡 omega=1 时跳过 uncond 网络调用（推理白送加速）

**文件**：`flow_loss.py:203-211`（`sample` 内的 sampler 循环）

**现状**：
```python
v_cond = self.net(x, t * 1000, x_exo_exp, y_exo_exp, prior_exp)
v_uncond = self.net(x, t * 1000, null_x_exo, null_y_exo, null_prior)
v_pred = v_uncond + self.omega * (v_cond - v_uncond)
```
当 `omega == 1.0`（默认值）时，结果等价于 `v_cond`，但每步仍跑了两次 `self.net`。

**改法**：
```python
v_cond = self.net(x, t * 1000, x_exo_exp, y_exo_exp, prior_exp)
if abs(self.omega - 1.0) < 1e-6:
    v_pred = v_cond
else:
    v_uncond = self.net(x, t * 1000, null_x_exo, null_y_exo, null_prior)
    v_pred = v_uncond + self.omega * (v_cond - v_uncond)
```

**预期**：`omega=1` 时推理时间约减半，结果数值上完全等价。`omega>1` 时无影响。

---

## 4. 🟡 Adam → AdamW，加 weight_decay

**文件**：`forecastingflow.py:144-145`

**现状**：
```python
def _init_optimizer(self,):
    return optim.Adam(self.model.parameters(), lr=self.config.lr)
```

**改法**：
```python
def _init_optimizer(self,):
    return optim.AdamW(
        self.model.parameters(),
        lr=self.config.lr,
        weight_decay=getattr(self.config, "weight_decay", 1e-4),
    )
```
并在 `MODEL_HYPER_PARAMS` 加 `"weight_decay": 1e-4`。

**预期**：Flow / 扩散类模型对 weight decay 敏感，AdamW 通常稳涨。论文若仅写 "Adam"，AdamW 通常被算同一类描述无需改写；如要严格匹配论文则跳过。

---

## 5. 🟡 Euler → Heun (二阶 RK) sampler

**文件**：`flow_loss.py:203-211`（同 #3 那段循环）

**现状**：forward Euler，`x = x + v * dt`。

**改法（与 #3 配合时，omega=1 路径每步 2 次 net 调用，与原 CFG 路径持平）**：
```python
for i in range(self.num_sampling_steps):
    t_val = i / self.num_sampling_steps
    t = torch.full((x.shape[0],), t_val, device=x.device)

    v1_cond = self.net(x, t * 1000, x_exo_exp, y_exo_exp, prior_exp)
    if abs(self.omega - 1.0) < 1e-6:
        v1 = v1_cond
    else:
        v1_unc = self.net(x, t * 1000, null_x_exo, null_y_exo, null_prior)
        v1 = v1_unc + self.omega * (v1_cond - v1_unc)

    x_pred = x + v1 * dt
    t2_val = (i + 1) / self.num_sampling_steps
    t2 = torch.full((x.shape[0],), t2_val, device=x.device)

    v2_cond = self.net(x_pred, t2 * 1000, x_exo_exp, y_exo_exp, prior_exp)
    if abs(self.omega - 1.0) < 1e-6:
        v2 = v2_cond
    else:
        v2_unc = self.net(x_pred, t2 * 1000, null_x_exo, null_y_exo, null_prior)
        v2 = v2_unc + self.omega * (v2_cond - v2_unc)

    x = x + 0.5 * (v1 + v2) * dt
```

**预期**：同 step 数下数值误差更小，常见小幅提升。属于数值方法选择，不动建模公式。

---

## 6. 🟢 训练加 EMA shadow，inference 用 EMA 权重

**文件**：`forecastingflow.py`（forecast_fit + 推理路径）

**思路**：
- 训练前初始化 `self.ema_model = copy.deepcopy(self.model)`，所有参数 detach。
- 每个 optimizer step 后做：`ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)`，`decay` 取 0.999 / 0.9999。
- EarlyStopping 改成对 ema_model 验证；inference 用 EMA 权重。
- 注意 buffer（如 dataset_prior）也要同步，最简单是每次 EMA 更新后 `ema_model.load_state_dict(...)` 同步 buffer。

**预期**：扩散 / flow 标准操作，通常涨幅最明显。论文方法零变化，只是工程封装。代价是显存 ~×2、训练步多一次拷贝。

---

## 推荐执行顺序

1 (必修) → 2 (必修) → 3 (白送) → 4 (轻松) → 6 (大涨幅) → 5 (看是否愿动 sampler)

如果不想冒任何风险，只做 1 + 2 + 3 就够了 —— 总改动 ~10 行。

---

## 验证建议

每改一条单独跑一次 baseline 比较，避免几条同时上线后归因不清。最少跑 3 个种子取均值，因为 flow / 扩散模型方差不小。
