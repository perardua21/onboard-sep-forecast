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
        idx = _nc_time(ds)
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
        idx = _nc_time(ds)
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
                # e_dif_* 는 ACE 몫이다. GOES 미분 전자는 정지궤도 방사선대 전자라
                # 이름을 따로 두고 모델 입력에는 넣지 않는다.
                out["e_mpsh_lo"] = d[:, lo]
                out["e_mpsh_hi"] = d[:, hi]
        if "EclipseFlag" in ds.variables:
            out["eclipse_flag"] = np.asarray(ds["EclipseFlag"].values)
        return out


#: 적분 양성자 임계 에너지 (MeV) → 공통 이름
SGPS_THRESHOLDS = {1: "p_gt1", 5: "p_gt5", 10: "p_gt10", 30: "p_gt30",
                   50: "p_gt50", 60: "p_gt60", 100: "p_gt100"}
_J2000 = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")


def _nc_time(ds) -> pd.DatetimeIndex:
    """시간 변수를 찾아 UTC 로. 제품마다 이름이 다르다 (time / L2_SciData_TimeStamp).

    xarray 가 단위를 풀었으면 datetime 으로 오고, 못 풀었으면 J2000 기준 초다.
    """
    name = next((v for v in ("time", "L2_SciData_TimeStamp") if v in ds.variables), None)
    if name is None:
        raise KeyError(f"시간 변수가 없다: {list(ds.variables)[:10]}")
    v = ds[name].values
    if np.issubdtype(v.dtype, np.datetime64):
        return pd.to_datetime(v, utc=True)
    return _J2000 + pd.to_timedelta(np.asarray(v, dtype="float64"), unit="s")


def integrate_spectrum(diff, e_eff_kev, thresholds_mev, e_top_mev=500.0, n_grid=400):
    """미분 플럭스(/keV) → 임계 에너지 이상 적분 플럭스(pfu).

    채널 유효 에너지 사이를 로그-로그 직선(국소 멱법칙)으로 잇고 e_top 까지
    적분한다. 채널이 겹치거나 사이가 비어 있어도 같은 방식으로 처리된다.
    e_top 이상은 따로 재는 적분 채널을 더해야 한다.

    diff : (n_rec, n_ch)   e_eff_kev : (n_ch,)
    """
    diff = np.asarray(diff, dtype="float64")
    e = np.asarray(e_eff_kev, dtype="float64")
    order = np.argsort(e)
    e, diff = e[order], diff[:, order]
    thr = np.asarray(sorted(thresholds_mev), dtype="float64") * 1000.0
    grid = np.logspace(np.log10(thr.min()), np.log10(e_top_mev * 1000.0), n_grid)
    lg, le = np.log(grid), np.log(e)
    out = np.full((diff.shape[0], thr.size), np.nan)
    for r in range(diff.shape[0]):
        ok = np.isfinite(diff[r]) & (diff[r] > 0)
        if ok.sum() < 2:
            continue
        x, y = le[ok], np.log(diff[r, ok])
        # 채널 범위 밖은 평평하게 두지 않고 양 끝 기울기로 연장한다.
        # 평평하게 두면 1 MeV 는 최대 30% 작게, 100 MeV 는 6% 크게 나온다.
        lo_s = (y[1] - y[0]) / (x[1] - x[0])
        hi_s = (y[-1] - y[-2]) / (x[-1] - x[-2])
        if lg[0] < x[0]:
            x, y = np.r_[lg[0], x], np.r_[y[0] + lo_s * (lg[0] - x[0]), y]
        if lg[-1] > x[-1]:
            x, y = np.r_[x, lg[-1]], np.r_[y, y[-1] + hi_s * (lg[-1] - x[-1])]
        j = np.exp(np.interp(lg, x, y))
        seg = 0.5 * (j[1:] + j[:-1]) * np.diff(grid)          # 사다리꼴, keV × /keV
        tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        out[r] = np.interp(thr, grid, tail)
    return out


def read_sgps_nc(path: Path) -> pd.DataFrame | None:
    """SGPS 양성자 (GOES-16 이후, 2020년부터).

    이 제품에는 ≥1·5·10·…·100 MeV 적분 플럭스가 **없다**. 13개 미분 채널과
    >500 MeV 적분 채널 하나뿐이다 (2021-10-28 파일로 확인). 그래서 미분
    스펙트럼을 적분해 만든다. ≥10 MeV 가 라벨 채널이므로 이 계산이 틀리면
    2020년 이후 사건이 통째로 틀린다. scripts/check_probe.py 로 확인할 것.

    센서 두 개(sensor_units)는 EPEAD 의 동·서와 같은 취지로 큰 값을 쓴다.
    """
    with _open_nc(path) as ds:
        need = ("AvgDiffProtonFlux", "DiffProtonEffectiveEnergy")
        if not all(v in ds.variables for v in need):
            print(f"  ! {Path(path).name}: 미분 양성자 변수가 없다. "
                  f"들어 있는 변수: {', '.join(list(ds.variables)[:20])}")
            return None
        idx = _nc_time(ds)
        diff = np.asarray(ds["AvgDiffProtonFlux"].values, dtype="float64")   # (rec, unit, ch)
        eeff = np.asarray(ds["DiffProtonEffectiveEnergy"].values, dtype="float64")
        if diff.ndim == 2:
            diff = diff[:, None, :]
        if eeff.ndim == 1:
            eeff = np.broadcast_to(eeff, (diff.shape[1], eeff.size))
        p11 = (np.asarray(ds["AvgIntProtonFlux"].values, dtype="float64")
               if "AvgIntProtonFlux" in ds.variables else None)
        if p11 is not None and p11.ndim == 1:
            p11 = p11[:, None]

        e_top = 500.0
        if "IntegralProtonEffectiveEnergy" in ds.variables:
            ie = np.asarray(ds["IntegralProtonEffectiveEnergy"].values, dtype="float64")
            e_top = float(np.nanmean(ie)) / 1000.0 if np.nanmax(ie) > 2000 else float(np.nanmean(ie))

        thr = sorted(SGPS_THRESHOLDS)
        per_unit = []
        for u in range(diff.shape[1]):
            J = integrate_spectrum(diff[:, u, :], eeff[u], thr, e_top)
            if p11 is not None:
                J = J + np.nan_to_num(p11[:, [u]], nan=0.0)
            per_unit.append(J)
        J = np.nanmax(np.stack(per_unit), axis=0)

        out = pd.DataFrame(J, index=idx, columns=[SGPS_THRESHOLDS[t] for t in thr])
        if p11 is not None:
            out["p_gt500"] = np.nanmax(p11, axis=1)
        return out


_DATE_IN_NAME = re.compile(r"_d(\d{8})")
_SAT_IN_NAME = re.compile(r"(?:^|_)g(\d{2})_")
_YEAR_IN_NAME = re.compile(r"(?:_d|_)((?:19|20)\d{2})\d{4}")


def policy_satellite(year: int) -> int:
    """연도별로 쓸 위성. scripts/fetch_data.py 의 satellite() 와 같아야 한다."""
    return 15 if year <= 2019 else (16 if year <= 2022 else 18)


def _policy_rank(path: Path) -> int:
    """같은 시각을 두 위성이 다 재고 있으면 정책 위성 파일을 먼저 읽게 한다.

    병합은 시각별 첫 유효값을 쓰므로 먼저 읽은 쪽이 남는다. 이름순으로만
    두면 폴더 이름에 따라 위성이 섞인다.
    """
    n = Path(path).name.lower()
    ms, my = _SAT_IN_NAME.search(n), _YEAR_IN_NAME.search(n)
    if not (ms and my):
        return 1
    return 0 if int(ms.group(1)) == policy_satellite(int(my.group(1))) else 2


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


def read_ace_hapi(path: Path) -> pd.DataFrame | None:
    """CDAWeb HAPI CSV (AC_H1_EPM). 머리말(#)의 JSON 에서 열 이름과 결측값을 읽는다.

    e_dif_lo = DE3 (0.103–0.175 MeV), e_dif_hi = DE4 (0.175–0.315 MeV).
    DE4 가 Núñez(2018)·REleASE 가 쓰는 준상대론적 전자 대역이다.
    """
    import json
    from io import StringIO
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    head = "\n".join(l[1:] for l in txt.splitlines() if l.startswith("#"))
    body = "\n".join(l for l in txt.splitlines() if l and not l.startswith("#"))
    if not body:
        return None
    names, fills = None, {}
    if head.strip():
        try:
            meta = json.loads(head)
            names = [p["name"] for p in meta["parameters"]]
            fills = {p["name"]: float(p["fill"]) for p in meta["parameters"]
                     if p.get("fill") not in (None, "", "null")}
        except Exception:                                    # noqa: BLE001
            names = None
    d = pd.read_csv(StringIO(body), header=None)
    if names is None or len(names) != d.shape[1]:
        names = ["Time"] + list(("DE1", "DE2", "DE3", "DE4"))[: d.shape[1] - 1]
    d.columns = names
    d.index = pd.to_datetime(d.iloc[:, 0], utc=True, errors="coerce")
    d = d.drop(columns=names[0])
    d = d[d.index.notna()]
    for c, f in fills.items():
        if c in d.columns:
            d[c] = d[c].where(d[c] != f)
    d = d.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=d.index)
    if "DE3" in d.columns:
        out["e_dif_lo"] = d["DE3"]
    if "DE4" in d.columns:
        out["e_dif_hi"] = d["DE4"]
    return out if len(out.columns) else None


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
            if n.startswith("ac_h1_epm"):
                return read_ace_hapi(path)
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
        present = s.notna()
        bad = present & ((s <= -1e30) | (s < 0))
        n_present = int(present.sum())
        # 자료가 없는 시각(공백)과 값은 있는데 버린 것(품질 탈락)을 나눠 센다.
        # 둘을 합쳐 '무효'라고 하면 기간 사이 공백까지 불량처럼 보인다.
        rep[c] = {"n": int(len(s)), "present": n_present, "invalid": int(bad.sum()),
                  "invalid_pct": round(100.0 * int(bad.sum()) / n_present, 3) if n_present else 0.0}
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
    files = sorted((p for p in src.rglob("*") if p.suffix in (".nc", ".csv")),
                   key=lambda p: (_policy_rank(p), p.name))
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
    # 관측률은 없는 채널을 0 으로 채우기 전에 센다
    coverage = {c: (float(df[c].notna().mean()) * 100 if c in df.columns else None)
                for c in ALL_COLUMNS}
    df = fill_missing(df)

    missing = [c for c in ALL_COLUMNS if c not in df.columns or df[c].isna().all()]
    if missing:
        print(f"  비어 있는 채널 {len(missing)}개: {', '.join(missing)}")
        print("  (선량률 2채널은 지구 자료에 없는 것이 정상이다)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _write_table(df, out_path)
    print(f"저장 {out_path}  {df.index[0]} ~ {df.index[-1]}  {len(df):,} 행")
    print("  채널별 관측률(전체 기간 대비) · 품질 탈락(관측값 중)")
    for c in ALL_COLUMNS:
        if c in DOSE:
            continue
        cov = coverage.get(c)
        if cov is None:
            print(f"    {c:10s} 없음")
            continue
        bad = report.get(c, {}).get("invalid_pct", 0.0)
        print(f"    {c:10s} 관측 {cov:5.1f}%   탈락 {bad:4.1f}%")
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
