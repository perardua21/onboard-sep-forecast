#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""달 궤도 선량 실측 자료(LRO/CRaTER) 내려받기.

    python scripts/fetch_dose.py --iswa
        iSWA 의 CRaTER 시간별 선량률(검출기별, 물 환산·달 표면 고도 보정)을
        2009~2026 전 기간 받는다. 수 MB.

    python scripts/fetch_dose.py --events
        S2(≥100 pfu) 이상 사건마다 시작 1일 전 ~ 2일 후의 1초 카운터를 받는다.
        시간별 선량률을 1분 단위로 나눌 때 쓴다. 사건당 약 70 MB.

    python scripts/fetch_dose.py --sec 2012-01-20 2012-01-31
        기간의 1초 카운터(secondary)만 받는다. 하루 17 MB.

CRaTER Level 2
  primary   : 입자 사건마다 D1~D6 에 남긴 에너지. 하루 15~19 GB 이고 SEP 중에는
              초당 약 1,200 사건으로 잘려 실려서 쓰지 않는다.
  secondary : 1초마다 검출기별 singles, good/reject/total 카운터. 하루 17 MB.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dose"
UA = {"User-Agent": "onboard-sep-forecast/1.0 (university research)"}

PDS = "https://pds-ppi.igpp.ucla.edu/data/lro-crater/"
COLL = {"sec": "data-der-sec"}
ISWA_HAPI = "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi/catalog"
HREF = re.compile(r'href="([^"?#][^"]*)"', re.I)

_log: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _log.append(s)


def _ctx():
    """윈도우 파이썬은 인증서 묶음이 없는 경우가 있다. certifi 가 있으면 그걸 쓴다."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                        # noqa: BLE001
        return None


CTX = _ctx()


def get(url: str, timeout: int = 60, headers: dict | None = None) -> bytes:
    with urlopen(Request(url, headers={**UA, **(headers or {})}), timeout=timeout,
                 context=CTX) as r:
        return r.read()


def head_size(url: str) -> int | None:
    try:
        req = Request(url, headers=UA, method="HEAD")
        with urlopen(req, timeout=60, context=CTX) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:                                        # noqa: BLE001
        return None


_LIST: dict[str, list[str]] = {}


def listing(url: str) -> list[str]:
    if url in _LIST:
        return _LIST[url]
    try:
        html = get(url).decode("utf-8", "replace")
    except Exception as e:                                   # noqa: BLE001
        say(f"  ! 목록 실패 {url}  ({e})")
        return []
    _LIST[url] = [m.group(1) for m in HREF.finditer(html) if not m.group(1).startswith((".", "/"))]
    return _LIST[url]


def stream(url: str, dest: Path, limit_mb: float | None = None) -> bool:
    """큰 파일은 조각으로 받아 바로 쓴다. 끊기면 .part 를 지우고 다시 받는다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        say(f"  = 이미 있음 {dest.name}")
        return True
    size = head_size(url)
    if limit_mb and size and size > limit_mb * 1e6:
        say(f"  · 건너뜀 {dest.name} ({size/1e6:.0f} MB > {limit_mb:.0f} MB)")
        return False
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urlopen(Request(url, headers=UA), timeout=300, context=CTX) as r, \
                open(part, "wb") as f:
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if size and got % (50 << 20) < (1 << 20):
                    print(f"    {got/1e6:.0f} / {size/1e6:.0f} MB", end="\r")
        part.replace(dest)
        say(f"  + {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as e:                                   # noqa: BLE001
        part.unlink(missing_ok=True)
        say(f"  ! 실패 {url}  ({e})")
        return False


def day_files(kind: str, d: date) -> list[str]:
    """그날의 파일 이름들 (버전이 여러 개면 모두)."""
    y, doy = d.year, d.timetuple().tm_yday
    names = listing(f"{PDS}{COLL[kind]}/{y}/")
    tag = f"CRAT_L2_{kind.upper()}_{y}{doy:03d}_"
    return sorted(n for n in names if n.startswith(tag))


ISWA_DS = ("CRaTER", "CRaTER_micro")


def _hapi_time(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_iswa(first: int = 2009, last: int | None = None, step_days: int = 7) -> None:
    """iSWA HAPI 시간별 선량률을 연도별 CSV 로.

    서버가 한 번에 긴 기간을 거절해서(400) 7일씩 나눠 받는다. 시작·끝은 서버가
    알려준 자료 범위 안으로 자른다.
    """
    base = ISWA_HAPI.rsplit("/", 1)[0]
    last = last or date.today().year
    for ds in ISWA_DS:
        info = json.loads(get(f"{base}/info?id={ds}"))
        names = [p["name"] for p in info["parameters"]]
        t_start = datetime.fromisoformat(info["startDate"].replace("Z", ""))
        t_stop = datetime.fromisoformat(info["stopDate"].replace("Z", ""))
        for y in range(max(first, t_start.year), min(last, t_stop.year) + 1):
            dest = OUT / "iswa" / f"{ds}_{y}.csv"
            if dest.exists() and dest.stat().st_size > 0 and y < t_stop.year:
                say(f"  = 이미 있음 {dest.name}")
                continue
            a, end = max(datetime(y, 1, 1), t_start), min(datetime(y + 1, 1, 1), t_stop)
            rows, fails = [], 0
            while a < end:
                b = min(a + timedelta(days=step_days), end)
                url = (f"{base}/data?id={ds}&time.min={_hapi_time(a)}&time.max={_hapi_time(b)}")
                try:
                    blob = get(url, timeout=300).decode("utf-8", "replace")
                    rows += [r for r in blob.splitlines() if r.strip()]
                except Exception as e:                       # noqa: BLE001
                    fails += 1
                    body = ""
                    if hasattr(e, "read"):
                        try:
                            body = e.read().decode("utf-8", "replace")[:300]
                        except Exception:                    # noqa: BLE001
                            pass
                    if fails <= 2:
                        say(f"  ! {ds} {a:%Y-%m-%d}~{b:%Y-%m-%d} 실패 ({e}) {body}")
                a = b
            if rows:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(",".join(names) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
                say(f"  + {dest.name}  {len(rows)}행" + (f"  (구간 {fails}개 실패)" if fails else ""))
            else:
                say(f"  ! {ds} {y} 받은 행 없음")


def event_days(min_peak_pfu: float = 100.0, before: int = 1, after: int = 2) -> list[date]:
    """처리된 GOES 자료에서 사건을 찾아, 정점이 min_peak_pfu 이상인 사건의 앞뒤 날짜."""
    sys.path.insert(0, str(ROOT / "src"))
    from oracle.config import load_config
    from oracle.ingest import load_dataset
    from oracle.labeling import detect_events
    cfg = load_config(ROOT / "configs" / "locked.yaml")
    path = cfg.path(cfg.get("data.path"))
    if not path.exists() and path.with_suffix(".csv").exists():
        path = path.with_suffix(".csv")
    df = load_dataset(path, cfg.cadence)
    p10 = df["p_gt10"]
    days: set[date] = set()
    n = 0
    for o, e in detect_events(p10):
        if p10.loc[o:e].max() < min_peak_pfu or o.year < 2009:
            continue
        n += 1
        for k in range(-before, after + 1):
            days.add((o + timedelta(days=k)).date())
    say(f"  정점 ≥{min_peak_pfu:g} pfu 사건 {n}건, 받을 날짜 {len(days)}일")
    return sorted(days)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--iswa", action="store_true", help="시간별 선량률 전 기간")
    g.add_argument("--events", action="store_true", help="S2 이상 사건 앞뒤 1초 카운터")
    g.add_argument("--sec", nargs=2, metavar=("START", "END"), help="기간의 secondary 만")
    a = ap.parse_args()
    if a.iswa:
        fetch_iswa()
        return 0
    if a.events:
        for d in event_days():
            for n in day_files("sec", d):
                if n.lower().endswith((".tab", ".xml")):
                    stream(f"{PDS}{COLL['sec']}/{d.year}/{n}", OUT / "crater" / n)
        return 0
    d, end = date.fromisoformat(a.sec[0]), date.fromisoformat(a.sec[1])
    while d <= end:
        for n in day_files("sec", d):
            stream(f"{PDS}{COLL['sec']}/{d.year}/{n}", OUT / "crater" / n)
        d += timedelta(days=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
