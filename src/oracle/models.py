# -*- coding: utf-8 -*-
"""모델 A - 30분 및 60분 안에 ≥10 pfu 를 넘을 확률 (출력 2개).

설계 기준 모델표
  아키텍처 : 1D CNN / TCN 백본 + sigmoid 출력부
  파라미터 : 백본 포함 5~15만
  연산량   : 20 MFLOPs / 추론

운용 요구이 요구하는 비교군
  지속성 · 단순 임계값 · 로지스틱 회귀 · 경량 신경망
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:                                      # noqa: BLE001
    HAS_TORCH = False


# ───────────────────────────────────────── 임계값
# ───────────────────────────────────────── 로지스틱 회귀
def fit_logreg(X, y, seed: int = 0, C: float = 1.0):
    """선형 기준선. 신경망이 선형보다 나은지를 같은 자료로 확인한다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=C, class_weight="balanced",
                           random_state=seed))
    clf.fit(X, y)
    return clf


def predict_proba(clf, X) -> np.ndarray:
    return clf.predict_proba(X)[:, 1]


# ───────────────────────────────────────── GBT (자료 점검용)
def fit_gbt(X, y, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed, class_weight="balanced")
    clf.fit(X, y)
    return clf


predict_gbt = predict_proba


# ───────────────────────────────────────── 연산 예산
def tcn_budget(n_feat: int, hidden: int = 32, n_blocks: int = 4,
               kernel_size: int = 7, n_heads: int = 2,
               window_steps: int = 12, flops_budget_m: float = 20.0) -> dict:
    """torch 없이 파라미터·연산량을 센다. 설계 기준 수치를 검산할 때 쓴다.

    합성곱 파라미터 = 입력채널 × 출력채널 × 커널 + 편향
    잔차 1×1 은 입력채널과 출력채널이 다를 때만 생긴다.
    """
    p, c = 0, n_feat
    for _ in range(n_blocks):
        p += c * hidden * kernel_size + hidden          # 1번 합성곱
        p += hidden * hidden * kernel_size + hidden     # 2번 합성곱
        if c != hidden:
            p += c * hidden + hidden                    # 잔차 1×1
        c = hidden
    p += hidden * n_heads + n_heads                     # 출력부
    conv_p = p - (hidden * n_heads + n_heads)
    flops = 2 * conv_p * window_steps + 2 * hidden * n_heads
    t_max = int(flops_budget_m * 1e6 // (2 * conv_p)) if conv_p else 0
    rf = receptive_field(n_blocks, kernel_size)
    return {"params": p, "flops": flops, "flops_m": flops / 1e6,
            "onboard_max_steps": t_max, "receptive_field": rf,
            "covers_window": rf >= window_steps}


def receptive_field(n_blocks: int, kernel_size: int) -> int:
    """팽창 1, 2, 4, ... 인 블록마다 합성곱 2개. 마지막 시점이 보는 과거 길이(점).

    이게 입력 길이보다 짧으면 창 앞부분은 모델에 닿지 않는다.
    """
    return 1 + 2 * (kernel_size - 1) * (2 ** n_blocks - 1)


def onboard_steps_for_budget(n_params: int, flops_budget_m: float = 20.0) -> int:
    """20 MFLOPs/추론 예산에 들어가는 입력 길이(점 수).

    합성곱 연산량 ≈ 2 × 파라미터 × 시간길이 이므로 T ≈ 예산 / (2 × 파라미터).
    센서는 1초로 재지만 60분 창을 1초 격자(3600점)로 넣으면 예산을 한참 넘는다.
    따라서 창을 재표본해서 넣어야 하고, 그 간격을 여기서 역산한다.
    """
    if n_params <= 0:
        return 0
    return int(flops_budget_m * 1e6 // (2 * n_params))


def resample_interval_s(n_params: int, window_min: int = 60,
                        flops_budget_m: float = 20.0) -> float:
    t = onboard_steps_for_budget(n_params, flops_budget_m)
    return float("inf") if t <= 0 else window_min * 60.0 / t


# ───────────────────────────────────────── TCN / 1D CNN
if HAS_TORCH:
    class _Chomp(nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x): return x[:, :, :-self.n] if self.n > 0 else x

    class _Block(nn.Module):
        """인과 팽창 합성곱 2개 + 잔차. 미래를 보지 않게 왼쪽만 패딩한다."""

        def __init__(self, c_in, c_out, k, dilation, dropout):
            super().__init__()
            pad = (k - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
                _Chomp(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
                _Chomp(pad), nn.ReLU(), nn.Dropout(dropout))
            self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

        def forward(self, x):
            r = x if self.down is None else self.down(x)
            return torch.relu(self.net(x) + r)

    class ModelA(nn.Module):
        """TCN 백본 + 출력부 2개 (30분, 60분). 설계 기준."""

        def __init__(self, n_feat: int, hidden: int = 32, n_blocks: int = 4,
                     kernel_size: int = 7, dropout: float = 0.2,
                     n_heads: int = 2, dilated: bool = True):
            super().__init__()
            blocks, c = [], n_feat
            for i in range(n_blocks):
                d = 2 ** i if dilated else 1
                blocks.append(_Block(c, hidden, kernel_size, d, dropout))
                c = hidden
            self.backbone = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden, n_heads)
            self.n_heads = n_heads

        def forward(self, x):                           # x: (B, T, F)
            h = self.backbone(x.transpose(1, 2))        # (B, C, T)
            return self.head(h[:, :, -1])               # (B, n_heads) - 마지막 시점만

    def make_model(kind: str, n_feat: int, cfg=None, n_heads: int = 2):
        g = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
        return ModelA(
            n_feat,
            hidden=int(g("model.channels_hidden", 32)),
            n_blocks=int(g("model.n_blocks", 4)),
            kernel_size=int(g("model.kernel_size", 7)),
            dropout=float(g("model.dropout", 0.2)),
            n_heads=n_heads,
            dilated=(kind != "cnn1d"))

    def fit_nn(X, y, model=None, n_feat=None, cfg=None, lr=2e-3, weight_decay=1e-4,
               epochs=20, batch_size=256, seed=0, device=None,
               Xva=None, yva=None, verbose=True, kind="tcn"):
        """y 는 (N,) 또는 (N, n_heads). 여러 horizon 을 한 번에 학습한다."""
        torch.manual_seed(seed); np.random.seed(seed)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        Y = y if y.ndim == 2 else y[:, None]
        if model is None:
            model = make_model(kind, n_feat or X.shape[2], cfg, n_heads=Y.shape[1])
        model = model.to(dev)

        pos = np.clip(Y.sum(axis=0).astype(float), 1.0, None)
        w = torch.tensor((len(Y) - pos) / pos, dtype=torch.float32, device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=w)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y.astype(np.float32))
        n = len(Xt)
        best_state, best_loss, patience = None, np.inf, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n); tot = 0.0
            for i in range(0, n, batch_size):
                sl = perm[i:i + batch_size]
                xb, yb = Xt[sl].to(dev), Yt[sl].to(dev)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(sl)
            msg = f"  epoch {ep+1}/{epochs}  train={tot/n:.4f}"
            if Xva is not None and len(Xva):
                Yv = yva if yva.ndim == 2 else yva[:, None]
                model.eval()
                with torch.no_grad():
                    vl = float(lossf(model(torch.from_numpy(Xva).to(dev)),
                                     torch.from_numpy(Yv.astype(np.float32)).to(dev)))
                msg += f"  valid={vl:.4f}"
                if vl < best_loss - 1e-4:
                    best_loss, patience, best_state = vl, 0, {
                        k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
            if verbose:
                print(msg)
            if patience >= 3:
                if verbose:
                    print("  early stopping")
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model.eval()

    @torch.no_grad()
    def predict_nn(model, X, batch_size=2048) -> np.ndarray:
        """(N, n_heads) 확률."""
        dev = next(model.parameters()).device
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(dev)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, model.n_heads), np.float32)

    def count_params(model) -> int:
        return sum(p.numel() for p in model.parameters())

    def flops_per_inference(model, window_steps: int) -> int:
        """합성곱·선형의 곱셈누산을 2 FLOPs 로 세고 시간축 길이를 곱한다."""
        f = 0
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                f += 2 * m.in_channels * m.out_channels * m.kernel_size[0] * window_steps
            elif isinstance(m, nn.Linear):
                f += 2 * m.in_features * m.out_features
        return f

    def budget_report(model, window_steps: int, cfg=None) -> dict:
        g = (lambda k, d: cfg.get(k, d)) if cfg is not None else (lambda k, d: d)
        lo, hi = g("model.param_budget", [50000, 150000])
        bud = float(g("model.flops_budget_m", 20))
        p = count_params(model)
        fl = flops_per_inference(model, window_steps)
        t_ob = onboard_steps_for_budget(p, bud)
        return {
            "파라미터": p,
            "파라미터 예산": f"{lo:,}~{hi:,}",
            "파라미터 충족": lo <= p <= hi,
            "연산량(지상 재현 T=%d)" % window_steps: f"{fl/1e6:.1f} MFLOPs",
            "연산량 예산": f"{bud:.0f} MFLOPs",
            "기내 입력 길이 상한": t_ob,
            "기내 창 재표본 간격": f"{resample_interval_s(p, int(g('label.window_min', 60)), bud):.0f} s",
        }
else:
    def _need_torch(*a, **k):
        raise RuntimeError("torch 가 없다. pip install torch 하거나 --model logreg 를 쓸 것")

    ModelA = None
    make_model = fit_nn = predict_nn = count_params = _need_torch
    flops_per_inference = budget_report = _need_torch
