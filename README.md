# onboard-sep-forecast

태양 양성자 사건(SEP) 조기경보 모델과, 그 성능을 공개 관측자료에서 사건 단위로
재는 검증 파이프라인. 위성 탑재를 상정한 경량 모델이라 파라미터와 연산량에
상한을 두고 설계했다.

모델은 최근 60분의 다중 센서 시계열을 읽고, **30분 및 60분 안에 ≥10 MeV 양성자
플럭스가 10 pfu(NOAA S1)에 도달할 확률**을 낸다.

## 설치와 실행

```bash
pip install -r requirements.txt
python run_tests.py
```

자료를 받기 전에 파이프라인만 확인하려면:

```bash
python scripts/run_all.py --synthetic --model logreg
```

실제 자료로 돌리려면 내려받기부터 한다.

```bash
python scripts/fetch_data.py --probe        # 이틀치 + 변수 목록
python scripts/fetch_data.py --all          # 전 구간 (수 시간)
python scripts/run_all.py --ingest data/raw
python scripts/run_all.py --model tcn --event 2017-09-10
```

설정은 `configs/locked.yaml`. 각 값이 무엇을 뜻하는지는 `docs/spec.md` 에 있다.

## 모델

| | |
|---|---|
| 입력 | 17채널 × 60분 (X선 2, 전자 5, 양성자 8, 선량률 2) |
| 출력 | 30분·60분 내 10 pfu 초과 확률 |
| 백본 | 1D CNN / TCN + sigmoid |
| 파라미터 | 60,002 (상한 150,000) |
| 연산량 | 20 MFLOPs/추론 예산 |

연산 예산이 입력 길이를 정한다. 60분 창을 1초 격자(3,600점)로 넣으면 예산을
한참 넘기므로 25초 간격 145점으로 재표본한다. 그 145점을 모델이 실제로 다
보도록 블록 수와 커널을 골랐다(수용영역 181점, 17.4 MFLOPs). `docs/spec.md` 참조.

17채널은 비행 설정이다. 지상 자료는 GOES 세대마다 채널 구성이 달라서, 학습·시험
구간 모두에서 관측된 채널만 남겨 쓴다. 무엇을 뺐는지는 실행할 때 출력된다.

양성자 채널은 **과거값만** 입력으로 쓴다. 라벨은 `t` 이후 구간에서 만들고,
사건이 진행 중인 구간은 학습과 평가에서 모두 뺀다. 이미 벌어진 일을 맞히는
것은 예측이 아니다.

## 평가

사건 수가 적고 양성 비율이 0.1% 수준이라 표본 단위 정확도는 쓸모가 없다.
아무 예보도 하지 않아도 99%를 넘기 때문이다. 사건 단위로 센다. 경보 하나는
"H분 안에 사건"이라는 주장으로 보고, 오래 켜진 경보는 H분 단위로 쪼개 센다.
그래야 경보를 계속 켜 두는 것으로 점수를 얻을 수 없다. 임계값도 같은 사건 단위
지표로 검증집합에서 고른다.

- 탐지율, 오경보비, CSI
- Brier 점수와 신뢰도 도표
- 선행시간의 평균·중앙값·하위 10백분위
- 95% 신뢰구간 (사건 블록 또는 기간 블록 복원추출)

신뢰구간을 블록 단위로 뽑는 이유는 이웃한 시각의 값이 서로 강하게 묶여 있어서다.
점 단위로 뽑으면 구간이 실제보다 좁게 나온다.

비교군은 지속성, 단순 임계값(X선/전자/양성자 각각), 2종 동의 규칙 게이트,
로지스틱 회귀, 경량 신경망이다. 제거 실험으로 채널 집합별 기여도를 본다.

### 기내 판정과 사용자 수신을 나눈다

판정이 빨라도 전달되지 않으면 경보가 없는 것과 같다. 저궤도 위성에서 지상
수신점이 보이는 시간은 한 바퀴 117.8분 중 12.43분뿐이다. `delivery.py` 가
궤도 위상을 균등 추첨해서 수신 선행시간의 분포를 내고, 수신이 사건 도달보다
늦은 경우를 따로 센다.

## 선량률 예측

SEP 시작 확률 대신 **10·30·60분 뒤 흡수선량률(µGy/min)** 을 직접 예측한다.
정점 ≥100 pfu(S2) 사건이 대상이다.

| | |
|---|---|
| 정답 | LRO/CRaTER D1·2 선량률 (달 궤도, 물 환산). 시간별 값을 1초 계수로 1분 단위로 나눔 |
| 입력 | 실측 선량률과 상승률, GOES 양성자로 계산한 선량률, 기존 GOES 입력 (최근 60분) |
| 출력 | log10 선량률 at t+10, 30, 60분 |
| 경보 | 예측이 10 또는 100 µGy/min 근처를 넘으면. 문턱은 학습 사건 안 교차검증으로 탐지율 70% 에 맞춘다 |
| 검증 | 사건 단위 7겹 교차검증 (28건) |

```bash
python scripts/fetch_dose.py --iswa          # 시간별 선량률 (수 MB)
python scripts/fetch_dose.py --events        # 사건 앞뒤 1초 계수 (사건당 약 70 MB)
python scripts/build_dose.py                 # data/processed/dose_1min.csv
python scripts/run_dose.py --cadence 1 --tcn --seeds 0,1,2
```

비교군은 지속성(지금 값 유지), 30분 상승률 외삽, 계산 선량(보정), 릿지, GBT, TCN.
경보는 30분 안에 이어진 것을 한 묶음으로 센다. 결과와 경과는 `docs/experiment_log.md`.

## 알려진 한계

- GOES-16 SGPS 양성자 자료는 2020년부터다. 학습 구간은 구세대 EPEAD 를 쓰게
  되므로 학습과 시험의 관측 기기가 다르다. 성능 차이에 기기 차이가 섞인다.
- EPEAD 는 >100 MeV 까지라 구세대 구간에서 `p_gt500` 이 결측이다.
- 선량률 2채널은 지상 자료로 채울 수 없다. 0 으로 두되 유무 플래그로 표시한다.
- 지상 재현은 5분 격자다. 10초 추론보다 선행시간이 한 격자만큼 짧게 나온다.
- 선량 실측은 달 궤도 LRO 한 기체의 한 검출기 쌍이다. 차폐가 다른 위치의 선량과는
  배율이 다르다. 사건이 28건이라 사건 단위 숫자는 1~2건에 크게 흔들린다.
- 합성 자료는 선행 신호를 규칙적으로 심어 둔 것이라 성능이 실제보다 훨씬 좋게
  나온다. 어떤 표에도 넣지 않는다.

## 구조

```
configs/locked.yaml     검증 설정
docs/spec.md            각 수치의 근거
docs/data_sources.md    무엇을 어디서 받는가
docs/data_dictionary.md 원본 변수명 → 공통 스키마
src/oracle/
  channels.py           채널 정의, 제거 실험용 채널 집합
  labeling.py           사건 탐지, 이중 horizon 라벨
  features.py           로그, 배경 대비 비율, 상승률
  datasets.py           창 생성, 시간순 분할, 표준화
  models.py             TCN, 로지스틱 회귀, 연산 예산 산식
  train.py              학습과 임계값 선택
  baselines.py          규칙 게이트와 단순 임계값
  predict.py            확률 → 경보
  evaluate.py           사건 단위 지표
  delivery.py           수신 선행시간
  experiments.py        표 생성
  ingest.py             원본 적재 (파일 여러 개를 시각 기준으로 병합)
  plots.py              그림
  dose.py               GOES 양성자 → 계산 선량률
  crater.py             CRaTER 시간별 선량률을 1분 단위로
  dosecast.py           선량률 예측 특징, 모델, 사건 단위 검증
scripts/fetch_data.py   공개 아카이브 내려받기
scripts/fetch_dose.py   CRaTER 선량 자료 내려받기
scripts/build_dose.py   1분 선량률 표 만들기
scripts/run_dose.py     선량률 예측 실행
tools/pytest_shim/      pytest 없는 환경용 대역
```

## 참고

- NOAA Space Weather Scales (S등급 정의)
- SEP Scoreboard / SPHINX 검증 틀 — 사건 시작·종료 정의와 표준 임계값
- Núñez & Paul-Peña (2018), *J. Space Weather Space Clim.* 8, A36 —
  X선과 준상대론적 전자를 쓴 예보 성능
- Posner (2007), *Space Weather* 5, S05001 — 전자 선행 신호 기반 예보

자료 출처는 `docs/data_sources.md` 참조.
