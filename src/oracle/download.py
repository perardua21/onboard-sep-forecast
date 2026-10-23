# -*- coding: utf-8 -*-
"""자료 수집 보조.

NOAA 아카이브 경로 규칙은 위성 세대마다 다르고 가끔 바뀐다.
템플릿을 그대로 믿지 말고 --probe 로 먼저 확인한 뒤 쓴다.
경로가 다르면 TEMPLATES 만 고치면 된다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

NGDC = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
NCEI_SCI = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science"
NCEI_AVG = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg"

#: 확인된 경로 (2026-10-19). 자세한 설명은 docs/data_sources.md
TEMPLATES = {
    # X선 - GOES-16/17/18/19. 변수: xrsa_flux(0.5–4 Å), xrsb_flux(1–8 Å)
    "xrs_goesr": f"{NGDC}/goes{{sat}}/l2/data/xrsf-l2-avg1m_science/{{y}}/{{m:02d}}/"
                 f"sci_xrsf-l2-avg1m_g{{sat}}_d{{y}}{{m:02d}}{{d:02d}}_v2-2-0.nc",
    # X선 - GOES-08~15 재처리판. 변수명이 GOES-R 과 같아 파서를 공유한다
    "xrs_legacy": f"{NCEI_SCI}/xrs/goes{{sat}}/xrsf-l2-avg1m_science/{{y}}/{{m:02d}}/"
                  f"sci_xrsf-l2-avg1m_g{{sat}}_d{{y}}{{m:02d}}{{d:02d}}_v1-0-0.nc",
    # 양성자 - GOES-16+ SGPS (1–500 MeV + >500 MeV 적분). 파일명 패턴 확인 필요
    "sgps_dir": f"{NGDC}/goes{{sat}}/l2/data/sgps-l2-avg5m/{{y}}/{{m:02d}}/",
    # 전자 - GOES-16+ MPS-HI (50–4000 keV + >2 MeV 적분). 파일명 패턴 확인 필요
    "mpsh_dir": f"{NGDC}/goes{{sat}}/l2/data/mpsh-l2-avg5m_science/{{y}}/{{m:02d}}/",
    # 양성자·전자 - GOES-13/15 EPEAD (CSV). 파일명 확인 필요
    "epead_dir": f"{NCEI_AVG}/{{y}}/{{m:02d}}/goes{{sat}}/csv/",
}

#: 사건 목록 - 라벨 교차 확인용
EVENT_LISTS = {
    "GSEP catalog (342 events, 1986-2017, Harvard Dataverse)":
        "https://doi.org/10.7910/DVN/DZYLHK",
    "NASA CDAW Major SEP Event List":
        "https://cdaw.gsfc.nasa.gov/CME_list/sepe/",
    "NOAA SWPC Solar Proton Events":
        "https://www.swpc.noaa.gov/",
}

#: 열 이름·단위를 눈으로 빨리 확인할 때 (학습에는 쓰지 않는다)
SWPC_JSON = "https://services.swpc.noaa.gov/json/goes/primary/"


def probe(url: str, timeout: int = 30) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        print(f"{r.status_code}  {url}")
        return r.status_code == 200
    except Exception as e:                              # noqa: BLE001
        print(f"ERR   {url}\n      {e}")
        return False


def fetch(url: str, dest: Path, manifest: Path | None = None, timeout: int = 180) -> Path:
    """원본 그대로 저장하고 manifest 에 기록한다. 재현성의 출발점이다."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    rec = {
        "url": url,
        "path": str(dest),
        "bytes": len(r.content),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "downloaded_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    }
    if manifest:
        manifest = Path(manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        rows.append(rec)
        manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"받음 {dest.name}  {rec['bytes']:,} B")
    return dest


def main():
    ap = argparse.ArgumentParser(description="NOAA 자료 수집 보조")
    ap.add_argument("--probe", action="store_true", help="URL 템플릿이 살아 있는지 확인")
    ap.add_argument("--url", type=str, help="단일 URL 내려받기")
    ap.add_argument("--dest", type=str, default="data/raw/manual.bin")
    ap.add_argument("--manifest", type=str, default="data/raw/manifest.json")
    a = ap.parse_args()
    if a.probe:
        for k, t in TEMPLATES.items():
            sat = 16 if ("goesr" in k or k in ("sgps_dir", "mpsh_dir")) else 15
            print(f"[{k}]  sat=g{sat}")
            probe(t.format(sat=sat, y=2017, m=9, d=10))
        print("\n사건 목록:")
        for k, v in EVENT_LISTS.items():
            print(f"  {k}\n    {v}")
        print(f"\n열 이름 빠른 확인: {SWPC_JSON}")
        print("자세한 설명: docs/data_sources.md")
        return
    if a.url:
        fetch(a.url, Path(a.dest), Path(a.manifest))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
