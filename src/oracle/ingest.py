# -*- coding: utf-8 -*-
"""원본 → 공통 스키마 (parse → QC → align).

실제 파일을 열어 확인한 변수명으로 매핑한다 (2026-10-23 확인).

GOES-R 세대 (netCDF)
  XRS      xrsf-l2-avg1m_science   xrsa_flux / xrsb_flux (+ xrsa_flag / xrsb_flag)
  전자     mpsh-l2-avg5m_science   AvgIntElectronFlux (>2 MeV), AvgDiffElectronFlux
  양성자   sgps-l2-avg5m           **2020년부터만 있다**

구세대 GOES-13/15 (NCEI CSV, 머리말 뒤 'data:' 부터가 본문)
  양성자   epead_cpflux_5m   ZPGT{1,5,10,30,50,60,100}{E,W} + _QUAL_FLAG
  전자     epead_e13ew_5m    E{1,2,3}{E,W}_COR_FLUX + _QUAL_FLAG
  고에너지 hepad_ap_5m       P8~P11_FLUX (약 330~700 MeV)

동서 두 검출기(E/W)는 **최댓값**을 쓴다. SEP 는 방향성이 있어 한쪽이 가려질 수
있고, 최댓값이 놓침을 줄이는 쪽이다. 이 선택은 docs/data_dictionary.md 에 적고
평균을 쓴 경우와 민감도를 비교한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .channels import ALL_COLUMNS, DOSE, XRAY_FLAG, fill_missing

TIME_COLS = ("time_tag", "time", "date", "datetime", "timestamp")

#: 구세대 CSV 양성자 적분 채널 → 공통 이름
EPEAD_PROTON = {"ZPGT1": "p_gt1", "ZPGT5": "p_gt5", "ZPGT10": "p_gt10",
                "ZPGT30": "p_gt30", "ZPGT50": "p_gt50", "ZPGT60": "p_gt60",
                "ZPGT100": "p_gt100"}
#: 구세대 CSV 전자 채널 → 공통 이름 (E1 >0.8, E2 >2, E3 >4 MeV)
EPEAD_ELECTRON = {"E1": "e_gt0p8", "E2": "e_gt2", "E3": "e_gt4"}


# CSV 공통
def _read_ncei_csv(path: Path) -> pd.DataFrame | None:
    """NCEI CSV. 머리말이 길고 'data:' 줄 다음부터가 본문이다."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^data:\s*$", txt, re.M)
    body = txt[m.end():] if m else txt
    try:
        from io import StringIO
        d = pd.read_csv(StringIO(body), skipinitialspace=True)
    except Exception:                                        # noqa: BLE001
        return None
    tc = next((c for c in d.columns if c.strip().lower() in TIME_COLS), None)
    if tc is None:
        return None
    d[tc] = pd.to_datetime(d[tc], utc=True, errors="coerce")
    d = d.dropna(subset=[tc]).set_index(tc)
    d.columns = [c.strip() for c in d.columns]
    return d


def _pair_max(d: pd.DataFrame, stem: str, suffix: str = "") -> pd.Series | None:
    """동(E)·서(W) 검출기 중 품질이 좋은 쪽의 최댓값.

    QUAL_FLAG 가 0 이 아닌 표본은 쓰지 않는다. 0 으로 채우지도 않는다.
    """
    got = []
    for side in ("E", "W"):
        col = f"{stem}{side}{suffix}"
        if col not in d.columns:
            continue
        s = pd.to_numeric(d[col], errors="coerce")
        q = f"{stem}{side}_QUAL_FLAG"
        if q in d.columns:
            s = s.where(pd.to_numeric(d[q], errors="coerce") == 0)
        got.append(s)
    if not got:
        return None
    return pd.concat(got, axis=1).max(axis=1, skipna=True)


def read_epead_protons(path: Path) -> pd.DataFrame | None:
    d = _read_ncei_csv(path)
    if d is None:
        return None
    out = {}
    for stem, name in EPEAD_PROTON.items():
        s = _pair_max(d, stem)
        if s is not None:
            out[name] = s
    return pd.DataFrame(out) if out else None


def read_epead_electrons(path: Path) -> pd.DataFrame | None:
    d = _read_ncei_csv(path)
    if d is None:
        return None
    out = {}
    for stem, name in EPEAD_ELECTRON.items():
        s = _pair_max(d, stem, "_COR_FLUX")
        if s is not None:
            out[name] = s
    return pd.DataFrame(out) if out else None


def read_hepad(path: Path) -> pd.DataFrame | None:
    """HEPAD P8~P11 (약 330~700 MeV). >500 MeV 적분의 대용이 아니라 참고값이다.

    EPEAD 는 >100 MeV 까지만 있어 p_gt500 을 채울 수 없다. 구세대 구간에서
    p_gt500 은 결측으로 두고 플래그로 표시한다. 없는 값을 만들지 않는다.
    """
    d = _read_ncei_csv(path)
    if d is None:
        return None
    cols = [c for c in d.columns if re.fullmatch(r"P(8|9|10|11)_FLUX", c)]
    if not cols:
        return None
    out = pd.DataFrame(index=d.index)
    for c in cols:
        q = c.replace("_FLUX", "_QUAL_FLAG")
        s = pd.to_numeric(d[c], errors="coerce")
        if q in d.columns:
            s = s.where(pd.to_numeric(d[q], errors="coerce") == 0)
        out[f"hepad_{c[:-5].lower()}"] = s
    return out


# netCDF
def _open_nc(path: Path):
    try:
        import xarray as xr
    except ImportError as e:                                 # noqa: BLE001
        raise SystemExit("netCDF 를 읽으려면 xarray 가 필요하다. "
                         "pip install xarray netcdf4") from e
    return xr.open_dataset(path)


def read_xrs_nc(path: Path) -> pd.DataFrame | None:
    with _open_nc(path) as ds:
        if "xrsb_flux" not in ds.variables:
            return None
        idx = pd.to_datetime(ds["time"].values, utc=True)
        out = pd.DataFrame(index=idx)
        for var, name, flag in (("xrsb_flux", "xrs_long", "xrsb_flag"),
                                ("xrsa_flux", "xrs_short", "xrsa_flag")):
            if var not in ds.variables:
                continue
            s = pd.Series(np.asarray(ds[var].values, dtype="float64"), index=idx)
            if flag in ds.variables:          # 0 이 아니면 쓰지 않는다
                s = s.where(np.asarray(ds[flag].values) == 0)
            out[name] = s
        return out if len(out.columns) else None


def read_mpsh_nc(path: Path) -> pd.DataFrame | None:
    """MPS-HI 전자. 망원경 5개 중 최댓값을 쓴다 (EPEAD 의 E/W 와 같은 취지)."""
    with _open_nc(path) as ds:
        if "AvgIntElectronFlux" not in ds.variables:
            return None
        idx = pd.to_datetime(ds["time"].values, utc=True)
        out = pd.DataFrame(index=idx)
        v = np.asarray(ds["AvgIntElectronFlux"].values, dtype="float64")
        out["e_gt2"] = np.nanmax(v, axis=1) if v.ndim > 1 else v   # E11 >2 MeV

        if "AvgDiffElectronFlux" in ds.variables:
            d = np.asarray(ds["AvgDiffElectronFlux"].values, dtype="float64")
            d = np.nanmax(d, axis=1) if d.ndim == 3 else d         # (time, chan)
            e = (np.asarray(ds["DiffElectronEffectiveEnergy"].values, dtype="float64")
                 if "DiffElectronEffectiveEnergy" in ds.variables else None)
            if e is not None and e.ndim > 1:
                e = np.nanmean(e, axis=0)
            if e is not None and d.shape[-1] == e.shape[-1]:
                # 저에너지·고에너지 대표 채널을 에너지로 고른다 (keV)
                lo = int(np.nanargmin(np.abs(e - 250.0)))
                hi = int(np.nanargmin(np.abs(e - 450.0)))
                out["e_dif_lo"] = d[:, lo]
                out["e_dif_hi"] = d[:, hi]
                out.attrs["e_dif_lo_keV"] = float(e[lo])
                out.attrs["e_dif_hi_keV"] = float(e[hi])
        if "EclipseFlag" in ds.variables:
            out["eclipse_flag"] = np.asarray(ds["EclipseFlag"].values)
        return out


def read_sgps_nc(path: Path) -> pd.DataFrame | None:
    """SGPS 양성자 (GOES-16 이후, 2020년~).

    적분 채널 변수명이 세대·버전마다 달라서 이름을 추측하지 않는다.
    'Int' 와 'Proton' 이 들어간 변수를 찾고, 임계 에너지 좌표로 열을 맞춘다.
    못 찾으면 무엇이 들어 있는지 알려주고 비운다.
    """
    want = {1: "p_gt1", 5: "p_gt5", 10: "p_gt10", 30: "p_gt30",
            50: "p_gt50", 60: "p_gt60", 100: "p_gt100", 500: "p_gt500"}
    with _open_nc(path) as ds:
        idx = pd.to_datetime(ds["time"].values, utc=True)
        cand = [v for v in ds.variables
                if "int" in v.lower() and "proton" in v.lower()
                and "flux" in v.lower() and "uncert" not in v.lower()]
        if not cand:
            print(f"  ! {Path(path).name}: 적분 양성자 변수를 못 찾았다.")
            print(f"    들어 있는 변수: {', '.join(list(ds.variables)[:25])}")
            return None
        var = sorted(cand, key=len)[0]
        arr = np.asarray(ds[var].values, dtype="float64")
        if arr.ndim == 3:                       # (time, telescopes, chans)
            arr = np.nanmax(arr, axis=1)
        if arr.ndim != 2:
            print(f"  ! {Path(path).name}: {var} 모양이 예상과 다르다 {arr.shape}")
            return None

        thr = None
        for v in ds.variables:                  # 임계 에너지 좌표 찾기
            if "int" in v.lower() and "energ" in v.lower():
                t = np.asarray(ds[v].values, dtype="float64")
                t = np.nanmean(t, axis=0) if t.ndim > 1 else t
                if t.shape[-1] == arr.shape[-1]:
                    thr = t / 1000.0 if np.nanmax(t) > 2000 else t   # keV→MeV
                    break
        out = pd.DataFrame(index=idx)
        if thr is None:
            print(f"  ! {Path(path).name}: {var} 의 임계 에너지 좌표를 못 찾았다. "
                  f"채널 수 {arr.shape[-1]}")
            return None
        for mev, name in want.items():
            j = int(np.nanargmin(np.abs(thr - mev)))
            if abs(thr[j] - mev) / mev < 0.25:   # 25% 안에 맞는 채널만
                out[name] = arr[:, j]
        return out if len(out.columns) else None


_DATE_IN_NAME = re.compile(r"_d(\d{8})")


def _check_dates(path: Path, d: pd.DataFrame | None) -> pd.DataFrame | None:
    if d is None or d.empty:
        return d
    m = _DATE_IN_NAME.search(Path(path).name)
    if not m:
        return d
    day = pd.Timestamp(m.group(1), tz="UTC")
    off = abs((d.index.min() - day).total_seconds()) / 86400
    if off > 2:
        raise ValueError(f"파일명 날짜 {day:%Y-%m-%d} 와 자료 시각 {d.index.min()} 이 "
                         f"{off:.0f}일 어긋난다. 시간 변수 단위를 확인할 것")
    return d


# 파일 종류 판별
def read_any(path: Path) -> pd.DataFrame | None:
    n = Path(path).name.lower()
    try:
        if n.endswith(".nc"):
            if "xrsf" in n:
                return _check_dates(path, read_xrs_nc(path))
            if "mpsh" in n:
                return _check_dates(path, read_mpsh_nc(path))
            if "sgps" in n:
                return _check_dates(path, read_sgps_nc(path))
            return None
        if n.endswith(".csv"):
            if "_5m_" not in n or "science" in n:
                return None
            if "cpflux" in n:
                return read_epead_protons(path)
            if "e13ew" in n:
                return read_epead_electrons(path)
            if "hepad_ap" in n:
                return read_hepad(path)
            return None
    except Exception as e:                                   # noqa: BLE001
        print(f"  ! {Path(path).name} 읽기 실패: {e}")
    return None


# QC · 정렬
def quality_control(df: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    """무효값 제거. 결측을 0 으로 채우지 않는다 - 0 은 실제 관측일 수 있다."""
    out = df.copy()
    rep = report if report is not None else {}
    for c in out.columns:
        if c.endswith(("_flag", "_ok")):
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        bad = s.isna() | (s <= -1e30) | (s < 0)
        rep[c] = {"n": int(len(s)), "invalid": int(bad.sum()),
                  "invalid_pct": round(float(bad.mean()) * 100, 3)}
        out[c] = s.where(~bad)
    return out


def align(df: pd.DataFrame, cadence_min: int = 5) -> pd.DataFrame:
    """UTC 공통 시간축. 원본보다 높은 해상도를 만들지 않는다."""
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    return out.resample(f"{cadence_min}min").mean()


def add_xray_flag(df: pd.DataFrame, eclipse: np.ndarray | None = None) -> pd.DataFrame:
    """X선 관측 가능 여부 플래그.

    지상 자료에는 달의 식이 없으므로 기본값은 1 이고 결측만 0 이다.
    궤도 식 패턴을 주면 그 구간을 0 으로 둔다.
    """
    out = df.copy()
    have = [c for c in ("xrs_long", "xrs_short") if c in out.columns]
    ok = (out[have].notna().all(axis=1).astype("float32") if have
          else pd.Series(0.0, index=out.index))
    if eclipse is not None:
        ok = ok.where(~pd.Series(eclipse, index=out.index), 0.0)
    out[XRAY_FLAG] = ok
    return out


# 묶기
def build_dataset(src_dir: Path, out_path: Path, cadence_min: int = 5) -> pd.DataFrame:
    """raw 폴더를 전부 읽어 processed 표 하나로."""
    src = Path(src_dir)
    files = sorted(p for p in src.rglob("*") if p.suffix in (".nc", ".csv"))
    parts: list[pd.DataFrame] = []
    used, skipped = [], []
    for p in files:
        d = read_any(p)
        if d is None or d.empty:
            skipped.append(p.name)
            continue
        parts.append(align(d, cadence_min))
        used.append(p.name)

    if not parts:
        raise SystemExit(
            f"읽은 파일이 없다: {src}\n"
            f"  scripts/fetch_data.py --probe 를 먼저 돌리고,\n"
            f"  docs/variable_inventory.txt 로 변수명을 확인할 것")

    print(f"읽음 {len(used)}개 / 건너뜀 {len(skipped)}개")
    # 열 방향으로 붙이면 같은 이름의 열(날짜만 다른 파일들)이 겹쳐 첫 파일만 남는다.
    # 행 방향으로 쌓고 시각별 첫 유효값을 고른다.
    df = pd.concat(parts, axis=0, sort=False).groupby(level=0).first()

    report: dict = {}
    df = align(quality_control(df, report), cadence_min)
    df = add_xray_flag(df)
    df = fill_missing(df)

    missing = [c for c in ALL_COLUMNS if c not in df.columns or df[c].isna().all()]
    if missing:
        print(f"  비어 있는 채널 {len(missing)}개: {', '.join(missing)}")
        print("  (선량률 2채널은 지구 자료에 없는 것이 정상이다)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _write_table(df, out_path)
    print(f"저장 {out_path}  {df.index[0]} ~ {df.index[-1]}  {len(df):,} 행")
    for c, r in sorted(report.items()):
        if r["invalid_pct"] > 1:
            print(f"  {c:12s} 무효 {r['invalid_pct']:.2f}%")
    return df


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    """parquet 로 저장하되 pyarrow 가 없으면 CSV 로 떨어진다."""
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path)
            return path
        except Exception as e:                               # noqa: BLE001
            path = path.with_suffix(".csv")
            print(f"  parquet 저장 실패({e.__class__.__name__}) → CSV 로 저장한다")
    df.to_csv(path)
    return path


def load_dataset(path: Path, cadence_min: int = 5) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet" and not p.exists() and p.with_suffix(".csv").exists():
        p = p.with_suffix(".csv")
    df = (pd.read_parquet(p) if p.suffix == ".parquet"
          else pd.read_csv(p, index_col=0, parse_dates=True))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index().resample(f"{cadence_min}min").mean()
    if XRAY_FLAG not in df.columns:
        df = add_xray_flag(df)
    return fill_missing(df)
