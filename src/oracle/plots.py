# -*- coding: utf-8 -*-
"""그림."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from matplotlib import font_manager                    # noqa: E402

from .channels import S1_PFU                           # noqa: E402

SKY = {"ink": "#0E2E42", "line": "#1B6CA8", "fill": "#CFE7F5",
       "accent": "#C2703D", "grid": "#D8E6EF", "mute": "#7FAFCB"}

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


def use_korean_font():
    for p in _FONT_CANDIDATES:
        try:
            font_manager.fontManager.addfont(p)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
            break
        except Exception:                               # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False


def _style(ax):
    ax.grid(True, color=SKY["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SKY["grid"])
    ax.tick_params(colors=SKY["ink"], labelsize=8)


def lead_time_hist(leads, path: Path, target_min: float = 30.0,
                   title: str = "10 pfu 도달 대비 기내 경보 선행시간 분포"):
    """선행시간 분포. 단일 숫자가 아니라 분포로 보여주는 것이 요점이다."""
    use_korean_font()
    leads = np.asarray(leads, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    if leads.size:
        bins = np.arange(0, max(120.0, leads.max() + 10), 10)
        ax.hist(leads, bins=bins, color=SKY["fill"], edgecolor=SKY["line"], lw=1.2)
        ax.axvline(target_min, color=SKY["accent"], lw=2, ls="--")
        ax.text(target_min + 1, ax.get_ylim()[1] * 0.92, f"목표 {target_min:.0f}분",
                color=SKY["accent"], fontsize=9, va="top")
        med = float(np.median(leads))
        ax.axvline(med, color=SKY["line"], lw=2)
        ax.text(med + 1, ax.get_ylim()[1] * 0.70, f"중앙값 {med:.0f}분",
                color=SKY["line"], fontsize=9)
        n_short = int((leads < target_min).sum())
        ax.text(0.99, 0.95, f"목표 미달 {n_short}건 / 전체 {leads.size}건",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=SKY["ink"])
    ax.set_xlabel("선행시간 [분]", color=SKY["ink"])
    ax.set_ylabel("사건 수", color=SKY["ink"])
    ax.set_title(title, color=SKY["ink"],
                 fontsize=11, fontweight="bold", loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def event_timeseries(df: pd.DataFrame, prob: np.ndarray, threshold: float,
                     onset: pd.Timestamp, path: Path,
                     before_h: int = 6, after_h: int = 12):
    """대표 사건의 X선·전자·양성자와 모델 확률, 경보 시점."""
    use_korean_font()
    w = df.loc[onset - pd.Timedelta(hours=before_h): onset + pd.Timedelta(hours=after_h)]
    p = pd.Series(prob, index=df.index).loc[w.index]
    rows = [("xrs_long", "X선 1-8 Å [W/m²]", SKY["line"]),
            ("e_gt2", "전자 >2 MeV", SKY["mute"]),
            ("p_gt10", "양성자 ≥10 MeV [pfu]", SKY["ink"])]
    fig, axes = plt.subplots(4, 1, figsize=(7.6, 7.4), dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for ax, (c, lab, col) in zip(axes, rows):
        ax.semilogy(w.index, w[c], color=col, lw=1.3)
        ax.set_ylabel(lab, fontsize=8.5, color=SKY["ink"])
        if c == "p_gt10":
            ax.axhline(S1_PFU, color=SKY["accent"], ls="--", lw=1.4)
            ax.text(w.index[2], S1_PFU * 1.4, "10 pfu (S1)",
                    color=SKY["accent"], fontsize=8)
        _style(ax)
    axes[3].plot(p.index, p.to_numpy(), color=SKY["line"], lw=1.4)
    axes[3].axhline(threshold, color=SKY["accent"], ls="--", lw=1.4)
    axes[3].fill_between(p.index, 0, 1, where=(p.to_numpy() >= threshold),
                         color=SKY["fill"], alpha=0.8, step="mid")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("모델 A 확률", fontsize=8.5, color=SKY["ink"])
    _style(axes[3])

    for ax in axes:
        ax.axvline(onset, color=SKY["accent"], lw=1.6)
    fired = p.index[p.to_numpy() >= threshold]
    title = f"{onset:%Y-%m-%d %H:%M} UTC 사건"
    if len(fired):
        for ax in axes:
            ax.axvline(fired[0], color=SKY["line"], lw=1.6, ls=":")
        title += f" - 경보 선행 {(onset - fired[0]).total_seconds()/60:.0f}분"
    axes[0].set_title(title, color=SKY["ink"], fontsize=11, fontweight="bold", loc="left")
    axes[3].set_xlabel("UTC", color=SKY["ink"])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def far_pod_curve(prob, y, path: Path, grid: int = 101):
    """임계값 sweep (EXP-04)."""
    use_korean_font()
    ok = np.isfinite(prob)
    prob, y = prob[ok], y[ok]
    ths = np.linspace(0.01, 0.99, grid)
    pods, fars = [], []
    for th in ths:
        pred = prob >= th
        tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        pods.append(tp / (tp + fn) if tp + fn else 0.0)
        fars.append(fp / (tp + fp) if tp + fp else 1.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.plot(ths, pods, color=SKY["line"], lw=2, label="탐지율 (표본 단위)")
    ax.plot(ths, fars, color=SKY["accent"], lw=2, ls="--", label="오경보율 (표본 단위)")
    ax.set_xlabel("임계값", color=SKY["ink"]); ax.set_ylabel("비율", color=SKY["ink"])
    ax.set_title("임계값에 따른 탐지율·오경보율", color=SKY["ink"],
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def event_pod_far_curve(curves: pd.DataFrame, chosen: float, path: Path,
                        target_far: float = 0.35, target_pod: float = 0.70):
    """사건 단위 탐지율-오경보비 곡선. 집합마다 선 하나, 고른 임계값에 점."""
    use_korean_font()
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    colors = [SKY["line"], SKY["accent"], SKY["mute"]]
    for k, (name, c) in enumerate(curves.groupby("집합", sort=False)):
        c = c.dropna(subset=["POD", "FAR"]).sort_values("threshold")
        col = colors[k % len(colors)]
        ax.plot(c["FAR"], c["POD"], color=col, lw=2, marker="o", ms=2.5, label=name)
        j = (c["threshold"] - chosen).abs().idxmin() if len(c) and chosen == chosen else None
        if j is not None:
            ax.plot(c.loc[j, "FAR"], c.loc[j, "POD"], "o", ms=8, mfc="white", mec=col, mew=2)
    ax.axvline(target_far, color=SKY["ink"], lw=1, ls=":")
    ax.axhline(target_pod, color=SKY["ink"], lw=1, ls=":")
    ax.text(target_far + 0.01, 0.03, f"오경보 상한 {target_far:.0%}", fontsize=8, color=SKY["ink"])
    ax.text(0.01, target_pod + 0.02, f"탐지율 목표 {target_pod:.0%}", fontsize=8, color=SKY["ink"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.set_xlabel("오경보비 (사건 단위)", color=SKY["ink"])
    ax.set_ylabel("탐지율 (사건 단위)", color=SKY["ink"])
    ax.set_title("임계값에 따른 탐지율과 오경보비 (흰 점 = 고른 임계값)", color=SKY["ink"],
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def reliability_diagram(rel: pd.DataFrame, path: Path):
    """신뢰도 도표.

    예측확률 0.7 이라고 말한 구간에서 실제로 70% 가 일어났는지 본다.
    대각선에서 멀면 확률값 자체를 그대로 쓰면 안 된다는 뜻이다.
    """
    use_korean_font()
    r = rel.dropna(subset=["pred_mean", "obs_freq"])
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(5.4, 5.6), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color=SKY["mute"], lw=1.2, ls="--")
    ax.plot(r["pred_mean"], r["obs_freq"], "o-", color=SKY["line"],
            lw=1.8, ms=5, mfc=SKY["fill"], mec=SKY["line"])
    ax.set_ylabel("실제 발생 빈도", color=SKY["ink"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("신뢰도 도표", color=SKY["ink"], fontsize=11,
                 fontweight="bold", loc="left")
    _style(ax)

    bx.bar(rel["bin_lo"] + 0.05, rel["n"].clip(lower=1), width=0.09,
           color=SKY["fill"], edgecolor=SKY["line"], lw=0.8)
    bx.set_yscale("log")
    bx.set_ylabel("표본 수", color=SKY["ink"], fontsize=8.5)
    bx.set_xlabel("예측 확률", color=SKY["ink"])
    _style(bx)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
