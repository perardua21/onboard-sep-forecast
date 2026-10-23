# 자료 출처와 내려받기

`configs/locked.yaml` 의 분할에 맞춰 무엇을 어디서 받아야 하는지 정리한다.

| 단계 | 기간 | 그 기간의 GOES 세대 |
|---|---|---|
| 학습 | 2010-01-01 ~ 2019-12-31 | GOES-13 / 15 (EPEAD) |
| 검증 | 2020-01-01 ~ 2022-12-31 | GOES-16 / 17 (SGPS·MPS-HI) |
| 잠금 시험 | 2023-01-01 ~ 2026-10-22 | GOES-16 / 18 / 19 |

**세대가 바뀌는 지점이 분할 경계와 거의 겹친다.** 학습은 구세대 기기로, 시험은
신세대 기기로 하게 된다는 뜻이다. 이건 우연이고, 성능 차이가 모델 때문인지
기기 때문인지 섞인다. 반드시 설계 기준 한계에 적고, 검증 구간을 두 세대에 걸쳐
확인한다.

---

## 0. 받는 순서 - 이 순서대로만 하면 된다

| 순서 | 할 일 | 걸리는 시간 |
|---|---|---|
| 1 | SWPC JSON 열어 열 이름·단위 눈으로 확인 | 1분 |
| 2 | **2017-09-10 하루치만** X선·양성자·전자 3종 받기 | 10분 |
| 3 | 받은 파일의 변수명을 출력해서 `ingest.RENAME` 수정 | 30분 ← **여기가 진짜 일** |
| 4 | 그 하루치로 `--ingest` 돌려 그림 확인 | 10분 |
| 5 | GSEP 사건 목록 받아 `detect_events()` 결과와 대조 | 1시간 |
| 6 | 전 기간 내려받기 | 수 시간 (스크립트로 밤새) |

**3번에서 막히면 받은 파일의 변수 목록만 나한테 보여주면 된다.** 아래 한 줄이면 나온다.

```python
import xarray as xr
print(xr.open_dataset("받은파일.nc"))      # netCDF
```
```python
import pandas as pd
print(pd.read_csv("받은파일.csv", nrows=5))  # CSV
```

---

## 1. X선 - 가장 쉽다. sunpy 한 줄

세대 구분이 필요 없다. 1986년부터 현재까지 `sunpy` 가 알아서 찾아온다.

```bash
pip install sunpy
```
```python
from sunpy.net import Fido, attrs as a

r = Fido.search(a.Time("2010-01-01", "2026-10-22"),
                a.Instrument("XRS"), a.Resolution("avg1m"))
Fido.fetch(r, path="data/raw/xrs/{file}")
```

하루치만 시험할 때는 날짜만 바꾼다.

```python
r = Fido.search(a.Time("2017-09-10", "2017-09-11"),
                a.Instrument("XRS"), a.Resolution("avg1m"),
                a.goes.SatelliteNumber(16))
```

### 직접 받고 싶으면

GOES-16 이후 (2017~현재)
```
https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/2017/09/
```
파일명 `sci_xrsf-l2-avg1m_g16_d20170910_v2-2-0.nc` · `goes16` 을 `goes18`, `goes19` 로 바꾸면 최신 위성

GOES-08 ~ 15 (1995~2020) - **재처리판**
```
https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes15/xrsf-l2-avg1m_science/2015/09/
```

구세대 재처리판도 **변수명이 신세대와 같다.** 그래서 X선은 파서 하나로 전 기간이 된다.

| 변수 | 뜻 | 우리 열 이름 |
|---|---|---|
| `xrsb_flux` | 1–8 Å | `xrs_long` |
| `xrsa_flux` | 0.5–4 Å | `xrs_short` |

하한은 1e-9 W/m². M1급은 `xrs_long` ≥ 1e-5 W/m².

---

## 2. 양성자 - 세대마다 다르다

**우리가 쓰는 8채널**: >1, >5, >10, >30, >50, >60, >100, >500 MeV 적분 플럭스 (pfu).
이 중 **>10 MeV 가 라벨을 만드는 채널**이다. 10 pfu 가 NOAA S1.

### 2010 ~ 2019 (학습 구간) - GOES-13 / 15, CSV

```
https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/avg/2015/09/goes15/csv/
```

이 폴더에 `g15_epead_cpflux_5m_*.csv` 류의 파일이 있다. 연도/월/위성만 바꾸면 된다.
GOES-13 은 `goes13`, 2010~2016 은 13·15 둘 다 있으니 하나를 주로 쓰고 결측만 보충한다.

CSV 머리말이 길다. `data:` 줄 다음부터가 본문이고, `ingest.parse_file` 이 처리한다.

### 2020 ~ 2026 (검증·시험 구간) - GOES-16 이후, netCDF

```
https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/sgps-l2-avg5m/2023/09/
```

SGPS (Solar and Galactic Proton Sensor), 1–500 MeV + >500 MeV 적분.
`goes16` → `goes18` / `goes19` 로 바꿔 최신 위성도 받는다.

**확인할 것**: SGPS 는 미분 채널이 기본이고 적분 채널이 별도 변수다. 어느 변수가
>10 MeV 적분이고 단위가 pfu 인지 파일을 열어 확인해야 한다. 미분을 적분으로
직접 바꾸려 하지 말 것 - 적분 변수가 파일 안에 들어 있다.

---

## 3. 전자 - 여기가 제일 헷갈린다

**필요한 건 GOES 전자가 아니다.** 설계 기준이 "SOHO 또는 ACE 전자"
라고 쓴 이유가 있다.

| 전자 | 에너지 | 선행 신호로서 |
|---|---|---|
| GOES EPEAD / MPS-HI | >0.8, >2, >4 MeV | 쓸 수는 있으나 약하다 |
| **ACE / EPAM** | **0.175–0.375 MeV** | **Núñez·REleASE 가 쓰는 채널. 이게 본류** |
| SOHO / EPHIN | 0.25–10.4 MeV | 1분 간격. REleASE 의 다른 입력 |

양성자보다 먼저 도착하는 "준상대론적 전자"는 수백 keV 대역이다. MeV 급 GOES
전자는 지구 자기권 안에서 다른 이유로도 오르내려서 오경보가 늘어난다.

### ACE EPAM 받는 곳

- ACE Science Center level 2: `https://izw1.caltech.edu/ACE/ASC/level2/`
- NASA CDAWeb (SPDF): `https://cdaweb.gsfc.nasa.gov/` - 데이터셋 `AC_K0_EPM`(키), `AC_H1_EPM`(5분 level 2)
- 파이썬: `pip install cdasws` 로 CDAWeb 을 코드에서 바로 받을 수 있다

```python
from cdasws import CdasWs
cdas = CdasWs()
status, data = cdas.get_data("AC_H1_EPM",
                             ["E_lo", "E_hi"],      # 변수명은 조회해서 확인
                             "2017-09-10T00:00:00Z", "2017-09-11T00:00:00Z")
```

### 17채널 중 전자 5채널을 무엇으로 채울지

설계 기준이 "전자 5"라고만 적어서 아직 정해지지 않았다. 아래가 제안이다.
정하고 나면 **실제 에너지 범위를 `docs/data_dictionary.md` 에 숫자로 적는다.**

| 코드의 열 이름 | 넣을 것 |
|---|---|
| `e_dif_lo` | ACE EPAM 0.175–0.315 MeV |
| `e_dif_hi` | ACE EPAM 0.315–0.580 MeV |
| `e_gt0p8` | GOES >0.8 MeV |
| `e_gt2` | GOES >2 MeV |
| `e_gt4` | GOES >4 MeV |

GOES 전자 위치
- 2010~2019: 양성자 CSV 와 같은 폴더의 `g15_epead_e13ew_*.csv`
- 2020~: `.../goes16/l2/data/mpsh-l2-avg5m_science/{YYYY}/{MM}/`

---

## 4. 선량률 2채널 - 지구 자료에는 없다

17채널 중 `dose_main`, `dose_backup` 은 **달 표면·달 궤도 선량계** 값이다.
지구 근처 관측 자료로는 채울 수 없다.

코드는 이 두 채널을 0 으로 채우고 `dose_main_ok = 0` 으로 **표시**한다.
없는 것을 있는 척하지 않는다. 제거 실험 표의 `no_dose (15채널)` 행이
실제로 돌릴 수 있는 설정이고, `fused17` 은 달 자료가 생긴 뒤의 설정이다.

달 환경 검증용
- LRO CRaTER: NASA PDS `https://pds-geosciences.wustl.edu/missions/lro/crater.htm`
- 기기 설명: `https://nssdc.gsfc.nasa.gov/nmc/experiment/display.action?id=2009-031A-06`

---

## 5. 사건 목록 - 라벨이 맞는지 대조하는 용도

### GSEP 카탈로그 학습 구간 검증용

Rotti, Aydin, Georgoulis, Martens, *Integrated Geostationary Solar Energetic
Particle Events Catalog: GSEP*. Harvard Dataverse **DOI 10.7910/DVN/DZYLHK**

- 342개 사건, 그중 246개가 >10 MeV 10 pfu 초과
- 1986–2017 (주기 22·23·24)
- GOES 적분 양성자 + 전자 E2·E3 + 양성자 P2~P7 시계열 조각이 **같이 들어 있다**
- 플레어·CME·활동영역 메타정보 포함

받아서 우리 `detect_events()` 가 뽑은 onset 과 대조한다. 크게 어긋나면 임계값·
지속조건·단위 중 하나가 틀린 것이다. **이 대조를 통과해야 그 뒤 숫자가 의미를 갖는다.**

한계: 2017년까지다. 잠금 시험 구간(2023~2026) 사건은 우리가 직접 뽑아야 한다.

### NOAA SWPC Solar Proton Events - 2018년 이후 확인용

`https://umbra.nascom.nasa.gov/SEP/` 또는 SWPC 의
"Solar Proton Events Affecting the Earth Environment" 목록.
잠금 시험 구간 사건 수를 확인하는 데 쓴다.

### 실시간 JSON - 포맷 감 잡기용 (학습에는 안 씀)

```
https://services.swpc.noaa.gov/json/goes/primary/integral-protons-plot-7-day.json
https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
```

브라우저에서 바로 열린다. 열 이름과 단위를 1분 만에 확인할 수 있다.

---

## 6. 분량

| 자료 | 1년 | 전 기간(2010–2026, 17년) |
|---|---|---|
| XRS avg1m | 수백 MB | 수 GB |
| 양성자 5분 | 수십 MB | 1 GB 안팎 |
| 전자 5분 | 수십 MB | 1 GB 안팎 |

하루 단위 파일이라 개수가 많다 (17년 × 365 × 3종 ≈ 1.9만 개).
`src/oracle/download.py` 의 `fetch()` 가 URL·크기·sha256·받은 날짜를 manifest 로
남기므로 재현성을 위해 그걸 쓴다.

---

## 7. 체크리스트

- [ ] XRS 변수명이 `xrsa_flux` / `xrsb_flux` 인가
- [ ] SGPS 에서 **>10 MeV 적분** 채널이 어느 변수이고 단위가 pfu 인가
- [ ] EPEAD CSV 의 양성자 열 이름
- [ ] ACE EPAM 전자 채널의 실제 에너지 경계 (숫자로 기록)
- [ ] MPS-HI 전자 채널의 실제 에너지 범위
- [ ] timestamp 가 UTC 인가, 기준 epoch 이 무엇인가
- [ ] 결측·품질 플래그를 0 이 아니라 NaN 으로 받고 있는가
- [ ] GOES-13/15 → 16+ 전환 시점을 `data_dictionary.md` 에 적었는가
- [ ] GSEP 사건과 `detect_events()` 결과가 맞는가
- [ ] 잠금 시험 구간(2023~2026) 사건이 몇 건인가 - 10건 미만이면 신뢰구간이 매우 넓어진다
