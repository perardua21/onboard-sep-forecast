#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공개 아카이브에서 자료를 내려받는다.

    python scripts/fetch_data.py --probe
        2017-09-10 하루치만 받고, 파일 안의 변수 이름을 뽑아
        docs/variable_inventory.txt 에 적는다. **이것부터 돌릴 것.**

    python scripts/fetch_data.py --all
        configs/locked.yaml 의 전 구간을 받는다. 수 시간 걸린다.

    python scripts/fetch_data.py --range 2017-09-01 2017-09-30
        기간을 직접 지정한다.

파일명 패턴을 미리 적어두지 않는다. 디렉터리 목록을 읽어서 날짜가 들어간
파일을 직접 고른다. NOAA 가 파일명 규칙을 바꿔도 스크립트가 따라간다.

필요한 것: 파이썬 표준 라이브러리만 있으면 내려받기는 된다.
변수 이름을 뽑으려면 xarray 가 필요하다 → pip install xarray netcdf4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "onboard-sep-forecast/1.0 (university research)"}

GOESR_BASE = ("https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
              "goes/goes{sat}/l2/data/")
GOESR = GOESR_BASE + "{prod}/{y:04d}/{m:02d}/"
LEGACY_XRS = ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor/"
              "access/science/xrs/goes{sat}/xrsf-l2-avg1m_science/{y:04d}/{m:02d}/")
LEGACY_CSV = ("https://www.ncei.noaa.gov/data/goes-space-environment-monitor/"
              "access/avg/{y:04d}/{m:02d}/goes{sat}/csv/")

#: 구세대 CSV 폴더에는 자기권 기기 자료까지 20개 넘게 들어 있다.
#: 한 달에 200 MB 라 10년이면 24 GB 가 되므로 쓰는 것만 받는다.
LEGACY_KEEP = ("epead_cpflux_5m",     # 양성자 적분 >1~>100 MeV
               "epead_e13ew_5m",      # 전자 E1/E2/E3
               "hepad_ap_5m")         # 고에너지 양성자 (참고값)

#: 연도 → 쓸 위성. 겹치는 해는 더 안정적인 쪽을 골랐다.
def satellite(y: int) -> int:
    """연도별 주 위성.

    학습 구간(2010~2019)은 GOES-15 하나로 맞춘다. 중간에 GOES-16 으로 바꾸면
    학습 자료 안에서 기기가 바뀌어 보정 차이가 섞인다. GOES-15 는 2020년 3월
    까지 운용됐다.
    """
    if y <= 2019:
        return 15
    if y <= 2022:
        return 16
    return 18


#: 구세대 주 위성 자료가 없는 달(예: GOES-15 운용 시작 전인 2010년 초)의 대안
LEGACY_FALLBACK = (15, 13)


def sources(d: date):
    """그 날짜에 받아야 할 (종류, [(디렉터리 URL, 파일명에 들어갈 날짜)...]) 목록.

    후보를 앞에서부터 시도해 처음 되는 것을 쓴다.
    """
    sat, y, m = satellite(d.year), d.year, d.month
    day = f"{d:%Y%m%d}"
    month = f"{d:%Y%m}"

    if sat == 15:
        xrs = [(LEGACY_XRS.format(sat=s_, y=y, m=m), day) for s_ in LEGACY_FALLBACK]
        par = [(LEGACY_CSV.format(sat=s_, y=y, m=m), month) for s_ in LEGACY_FALLBACK]
        if y >= 2017:                           # GOES-15 가 비면 GOES-16 X선으로
            prod = resolve_product(16, "xrs", ("avg1m",))
            if prod:
                xrs.append((GOESR.format(sat=16, prod=prod, y=y, m=m), day))
        return [("xrs", xrs), ("particles", par)]

    out = []
    for kind, keyword, prefer in (
            ("xrs", "xrs", ("avg1m",)),
            ("protons", "sgps", ("avg5m", "avg1m")),
            ("electrons", "mpsh", ("avg5m", "avg1m"))):
        cands = []
        prod = resolve_product(sat, keyword, prefer)
        if prod:
            cands.append((GOESR.format(sat=sat, prod=prod, y=y, m=m), day))
        if y == 2020:                           # GOES-15 는 2020년 3월까지
            if kind == "xrs":
                cands.append((LEGACY_XRS.format(sat=15, y=y, m=m), day))
            else:
                cands.append((LEGACY_CSV.format(sat=15, y=y, m=m), month))
        if cands:
            out.append((kind, cands))
    return out


# 네트워크
def get(url: str, timeout: int = 60) -> bytes:
    with urlopen(Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


HREF = re.compile(rb'href="([^"?][^"]*)"', re.I)


def list_dir(url: str, dirs: bool = False, quiet: bool = False) -> list[str]:
    """디렉터리 목록에서 이름을 뽑는다. dirs=True 면 하위 폴더만."""
    try:
        html = get(url)
    except Exception as e:                                   # noqa: BLE001
        if not quiet:
            print(f"  ! 디렉터리를 못 열었다: {url}\n    {e}")
        return []
    names = []
    for m in HREF.finditer(html):
        n = m.group(1).decode("utf-8", "replace")
        if n.startswith(("..", "?", "#", "/")):
            continue
        is_dir = n.endswith("/")
        if is_dir != dirs:
            continue
        names.append(n.rstrip("/") if dirs else n)
    return names


_PRODUCT_CACHE: dict[tuple[int, str], str | None] = {}


def resolve_product(sat: int, keyword: str, prefer=("avg5m", "avg1m")) -> str | None:
    """l2/data 아래에서 keyword 가 든 제품 폴더 이름을 직접 찾는다.

    NOAA 가 제품 폴더 이름을 바꾸거나 _science 접미사 유무가 달라도
    목록을 읽어 맞춘다. 이름을 추측해 적어두면 404 가 난다.
    """
    key = (sat, keyword)
    if key in _PRODUCT_CACHE:
        return _PRODUCT_CACHE[key]
    base = GOESR_BASE.format(sat=sat)
    folders = [d for d in list_dir(base, dirs=True, quiet=True)
               if keyword in d.lower()]
    pick = None
    for want in prefer:
        cands = [d for d in folders if want in d]
        if cands:
            # _science 가 붙은 쪽을 우선한다 (보정본)
            cands.sort(key=lambda d: (0 if d.endswith("_science") else 1, len(d)))
            pick = cands[0]
            break
    if pick is None and folders:
        pick = sorted(folders)[0]
    if pick:
        print(f"  · {keyword} 제품 폴더: {pick}")
    else:
        print(f"  ! {keyword} 제품 폴더를 못 찾았다. {base} 에 있는 폴더: "
              f"{', '.join(list_dir(base, dirs=True, quiet=True)) or '(목록 없음)'}")
    _PRODUCT_CACHE[key] = pick
    return pick


def diagnose(url: str) -> None:
    """월 폴더가 없을 때, 위로 올라가며 무엇이 있는지 알려준다."""
    parts = url.rstrip("/").split("/")
    for up, label in ((2, "연도"), (1, "월")):
        parent = "/".join(parts[:-up]) + "/"
        have = list_dir(parent, dirs=True, quiet=True)
        if have:
            shown = ", ".join(have[:14]) + (" ..." if len(have) > 14 else "")
            print(f"    · {parent} 안의 {label} 폴더: {shown}")
            if up == 2 and have:
                print(f"    · 이 제품은 {have[0]} 년부터 있다")
            return


def download(url: str, dest: Path) -> dict | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = 이미 있음  {dest.name}")
        return {"url": url, "path": str(dest.relative_to(ROOT)),
                "bytes": dest.stat().st_size, "skipped": True}
    for attempt in range(3):
        try:
            blob = get(url, timeout=180)
            dest.write_bytes(blob)
            print(f"  + {dest.name}  ({len(blob)/1e6:.1f} MB)")
            return {"url": url, "path": str(dest.relative_to(ROOT)),
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "fetched": time.strftime("%Y-%m-%dT%H:%M:%S")}
        except Exception as e:                               # noqa: BLE001
            if attempt == 2:
                print(f"  ! 실패  {url}\n    {e}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def fetch_day(d: date, manifest: list) -> None:
    print(f"\n[{d:%Y-%m-%d}]  GOES-{satellite(d.year)}")
    for kind, candidates in sources(d):
        got = False
        for i, (url, pattern) in enumerate(candidates):
            names = [n for n in list_dir(url, quiet=True) if pattern in n]
            if "/access/avg/" in url:          # 구세대 폴더는 쓰는 것만
                names = [n for n in names if any(k in n for k in LEGACY_KEEP)]
            if not names:
                if i == 0:
                    print(f"  ! {kind}: {url} 에 {pattern} 이 없다")
                    diagnose(url)
                    if len(candidates) > 1:
                        print("    → 구세대 자료로 넘어간다")
                continue
            for n in names:
                rec = download(urljoin(url, n), RAW / kind / n)
                if rec:
                    rec["kind"] = kind
                    manifest.append(rec)
            got = True
            break
        if not got:
            print(f"  ! {kind}: 받을 수 있는 자료를 못 찾았다")


# 변수 이름 뽑기
def inventory(out_path: Path) -> None:
    """받은 파일에서 변수 이름·단위·차원을 뽑아 한 파일에 적는다."""
    lines = ["# 받은 파일의 변수 목록",
             "# 이 파일을 그대로 보내면 파서를 맞출 수 있다.", ""]
    try:
        import xarray as xr
        has_xr = True
    except ImportError:
        has_xr = False
        lines.append("!! xarray 가 없어 netCDF 내부를 못 읽었다.")
        lines.append("!! pip install xarray netcdf4  후 다시 돌릴 것.")
        lines.append("")

    for f in sorted(RAW.rglob("*")):
        if not f.is_file():
            continue
        lines.append("=" * 70)
        lines.append(f"{f.relative_to(ROOT)}   ({f.stat().st_size/1e6:.2f} MB)")
        lines.append("=" * 70)
        if f.suffix == ".nc" and has_xr:
            try:
                with xr.open_dataset(f) as ds:
                    lines.append(f"차원: {dict(ds.sizes)}")
                    lines.append("")
                    for name, v in ds.variables.items():
                        unit = v.attrs.get("units", "")
                        desc = (v.attrs.get("long_name")
                                or v.attrs.get("description", ""))[:60]
                        lines.append(f"  {name:<28} {str(v.dims):<24} "
                                     f"{unit:<12} {desc}")
            except Exception as e:                           # noqa: BLE001
                lines.append(f"  읽기 실패: {e}")
        elif f.suffix in (".csv", ".txt"):
            try:
                head = f.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
                lines += ["  " + h[:160] for h in head]
            except Exception as e:                           # noqa: BLE001
                lines.append(f"  읽기 실패: {e}")
        else:
            lines.append("  (내용 생략)")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n변수 목록을 적었다 → {out_path}")
    print("이 파일을 그대로 보내면 된다.")


# 진입점
def daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser(description="공개 아카이브에서 자료 내려받기")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe", action="store_true",
                   help="2017-09-10 하루치만 받고 변수 목록을 뽑는다 (먼저 이것부터)")
    g.add_argument("--all", action="store_true",
                   help="configs/locked.yaml 의 전 구간")
    g.add_argument("--range", nargs=2, metavar=("시작", "끝"),
                   help="YYYY-MM-DD YYYY-MM-DD")
    ap.add_argument("--inventory-only", action="store_true",
                    help="내려받지 않고 이미 있는 파일의 변수 목록만 뽑는다")
    a = ap.parse_args()

    inv = ROOT / "docs" / "variable_inventory.txt"
    if a.inventory_only:
        inventory(inv)
        return

    if a.probe:
        days = [date(2017, 9, 10), date(2021, 10, 28)]
        print("두 날짜만 받는다. 둘 다 대형 SEP 사건이 있던 날이다.")
        print("  2017-09-10 (GLE 72)  구세대 GOES-15 CSV 포맷")
        print("  2021-10-28 (GLE 73)  신세대 GOES-16 SGPS netCDF 포맷")
        print("양성자 SGPS 자료가 2020년부터라 두 포맷을 다 확인해야 한다.")
    elif a.range:
        days = list(daterange(date.fromisoformat(a.range[0]),
                              date.fromisoformat(a.range[1])))
    else:
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "locked.yaml").read_text(encoding="utf-8"))
        s = cfg["split"]
        days = list(daterange(date.fromisoformat(s["train_start"]),
                              date.fromisoformat(s["test_end"])))
        print(f"{s['train_start']} ~ {s['test_end']}  ({len(days):,}일)")
        print("수 시간 걸린다. 중간에 끊겨도 다시 돌리면 받은 건 건너뛴다.")

    manifest = []
    try:
        for i, d in enumerate(days, 1):
            fetch_day(d, manifest)
            if len(days) > 30 and i % 30 == 0:
                print(f"  ... {i}/{len(days)}일")
    except KeyboardInterrupt:
        print("\n중단됨. 받은 것까지 manifest 에 남긴다.")

    if manifest:
        mp = RAW / "manifest.jsonl"
        with open(mp, "a", encoding="utf-8") as f:
            for r in manifest:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{len(manifest)}개 파일. 기록 → {mp.relative_to(ROOT)}")

    if a.probe:
        inventory(inv)


if __name__ == "__main__":
    sys.exit(main())
