import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

def _ts_rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    windows = x.unfold(1, window, 1)
    mean = windows.mean(dim=-1)
    pad = torch.zeros((x.shape[0], window - 1), device=x.device)
    return torch.cat([pad, mean], dim=1)

def _ts_rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(x)
    windows = x.unfold(1, window, 1)
    std = windows.std(dim=-1, unbiased=False)
    pad = torch.zeros((x.shape[0], window - 1), device=x.device)
    return torch.cat([pad, std], dim=1)

def _ts_rolling_zscore(x: torch.Tensor, window: int) -> torch.Tensor:
    mean = _ts_rolling_mean(x, window)
    std = _ts_rolling_std(x, window)
    return (x - mean) / (std + 1e-6)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x, 1), _ts_delay(x, 2))), 1),
    ('TS_MEAN_30', lambda x: _ts_rolling_mean(x, 30), 1),
    ('TS_MEAN_90', lambda x: _ts_rolling_mean(x, 90), 1),
    ('TS_MEAN_180', lambda x: _ts_rolling_mean(x, 180), 1),
    ('TS_STDDEV_30', lambda x: _ts_rolling_std(x, 30), 1),
    ('TS_STDDEV_90', lambda x: _ts_rolling_std(x, 90), 1),
    ('TS_STDDEV_180', lambda x: _ts_rolling_std(x, 180), 1),
    ('TS_ZSCORE_30', lambda x: _ts_rolling_zscore(x, 30), 1),
    ('TS_ZSCORE_90', lambda x: _ts_rolling_zscore(x, 90), 1),
    ('TS_ZSCORE_180', lambda x: _ts_rolling_zscore(x, 180), 1)
]