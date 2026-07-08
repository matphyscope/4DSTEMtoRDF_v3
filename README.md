# fourdstem

범용 **4D-STEM 처리·분석 툴킷**. 변환(conversion)뿐 아니라 분석까지 — RDF/PDF `G(r)`,
방위각 적분 `I(q)`, virtual imaging(BF/ADF/DPC), **NMF/PCA 분해**, **Bragg peak 검출·마스킹**,
그리고 **in-situ 시계열**(온도/시간 series) 처리를 하나의 API로 다룹니다.
데이터 **하나**만 있어도 되고, **여러 개(in-situ)** 여도 됩니다.

`py4DSTEM`처럼 함수/클래스를 카테고리별 폴더(subpackage)로 묶어 두어 필요한 조각만 골라 쓸 수 있게 설계했습니다.
기존 `batch_dm4_to_rdf.py` 스크립트의 RDF 파이프라인이 `fourdstem.analysis.rdf` 안으로 재구성되어 들어가 있습니다.

## 구조

```
fourdstem/
├── io/           데이터 입출력 + DataCube 컨테이너
│   ├── datacube.py     DataCube, Calibration
│   ├── readers.py      load / load_dm4 / load_generic / from_array
│   └── writers.py      npz 저장·복원 (결과 & DataCube)
├── preprocess/   전처리
│   ├── calibration.py  q_per_px 캘리브레이션, 픽셀↔q 변환
│   ├── center.py       beam center (Friedel 대칭 / center-of-mass)
│   ├── masks.py        beam stopper, Bragg peak 마스크, virtual detector(disk/annulus/wedge)
│   └── transform.py    mean pattern, crop/bin, polar 변환
├── analysis/     분석
│   ├── azimuthal.py    I(q) 방위각 적분, 방위각 분산(anisotropy)
│   ├── virtual_image.py BF / ADF / center-of-mass(DPC) 맵
│   ├── peaks.py        1D 프로파일 peak 검출·refine·centroid
│   ├── decomposition.py NMF / PCA 분해, 재구성(denoise)
│   └── rdf.py          scattering factor, reduction φ(q), sine FT → G(r)
├── insitu/       in-situ 시계열
│   ├── series.py       Series (파일/큐브에서 프레임 정렬 로드)
│   └── tracking.py     프레임 간 peak 위치/세기/적분 추적
├── viz/          시각화 (matplotlib, lazy import)
└── utils/        공용 헬퍼
```

## 설치

### 방법 A — conda 환경 새로 만들기 (권장)

저장소 루트에서:

```bash
conda env create -f environment.yml   # 'fourdstem' 환경 생성 (+ editable 설치)
conda activate fourdstem
python -c "import fourdstem as fds; print(fds.__version__)"   # 확인
```

환경을 다시 만들려면(깨끗하게):

```bash
conda env remove -n fourdstem
conda env create -f environment.yml
```

`environment.yml`을 수정한 뒤 반영:

```bash
conda env update -f environment.yml --prune
```

### 방법 B — conda 환경만 만들고 수동 설치

```bash
conda create -n fourdstem python=3.11 -y
conda activate fourdstem
conda install -c conda-forge numpy scipy scikit-learn matplotlib jupyterlab ncempy -y
pip install -e .                       # fourdstem 패키지 (editable)
# 선택: 대체 리더 / 정량 scattering factor
# conda install -c conda-forge hyperspy abtem -y
```

### 방법 C — pip만 (extras 사용)

```bash
pip install -e .              # 핵심(numpy, scipy)만
pip install -e ".[all]"       # + sklearn, ncempy, hyperspy, abtem, matplotlib
```

선택적 의존성 그룹: `ml`(NMF/PCA), `io`(dm4 읽기), `sf`(정량 scattering factor), `viz`(플롯).
핵심은 numpy/scipy만으로 동작하고, 없는 기능은 필요할 때 친절한 안내를 띄웁니다.

> **Jupyter에서 커널이 안 보이면**: `python -m ipykernel install --user --name fourdstem`
> 실행 후 노트북에서 `fourdstem` 커널을 선택하세요.

## 빠른 시작

### 1) 데이터 하나 분석

```python
import fourdstem as fds

cube = fds.load("scan.dm4")               # → DataCube (2D/3D/4D 모두 동일 API)
pattern = fds.to_pattern(cube)            # 평균 회절 패턴

# beam stopper 마스크 + Friedel 대칭으로 center 찾기
stopper = fds.beam_stopper_mask(pattern)
(cx, cy), fried = fds.find_center(pattern, stopper)

# Bragg 스팟 검출 후 마스킹 → 비정질 halo만 남김
bragg = fds.bragg_peak_mask(pattern, center=(cx, cy), sigma=5)
mask  = fds.combine_masks(stopper, bragg)

q, Iq = fds.azimuthal_integrate(pattern, (cx, cy), cube.calibration.q_per_px, mask)

# 환원 밀도 함수 G(r)
cfg = fds.RDFConfig(composition={"Si": 1, "O": 2}, q_int_min=0.8, q_int_max=12.0)
rdf = fds.pattern_to_rdf(pattern, cube.calibration.q_per_px, cfg,
                         center=(cx, cy), mask=bragg)
r1, _ = fds.first_peak_position(rdf.r, rdf.Gr, 1.3, 2.2)   # 첫 배위 거리

# 4D scan이면 virtual image
bf  = fds.bright_field(cube, center=(cx, cy))
adf = fds.annular_dark_field(cube, center=(cx, cy))
```

### 2) in-situ 시계열 (예: 비정질 SiOx 온도 series)

```python
import fourdstem as fds

cfg = fds.RDFConfig(composition={"Si": 1, "O": 2},
                    q_int_min=0.8, q_int_max=12.0, r_min=1.10)  # series 전체 LOCK

series = fds.Series.from_files("data/*.dm4")       # 파일명에서 온도(예: _450K_) 자동 파싱
rdfs = series.map(lambda f: fds.pattern_to_rdf(f.pattern, f.q_per_px, cfg))

profiles = [(r.r, r.Gr) for r in rdfs]
track = fds.track_peak(profiles, series.coordinates(), (1.3, 2.2))
# track["coord"], track["position"] → 온도에 따른 첫 배위 거리 변화
```

### 3) NMF로 Bragg / 비정질 성분 분리

```python
import fourdstem as fds

cube = fds.load("scan.dm4")
mean = cube.mean_dp()
stopper = fds.beam_stopper_mask(mean)

result = fds.nmf_decompose(cube, n_components=4, mask=stopper)
# result.components : (k, Qy, Qx)  각 성분의 회절 패턴
# result.loadings   : (k, Ry, Rx)  각 성분이 실공간 어디에 있는지

peaks = fds.detect_bragg_peaks(cube.max_dp(), center=fds.find_center(mean, stopper)[0],
                               q_per_px=cube.calibration.q_per_px)
```

`examples/` 폴더에 위 세 흐름을 그대로 실행 가능한 스크립트로 넣어 두었습니다:
`single_dataset.py`, `insitu_series_rdf.py`, `nmf_bragg_separation.py`.

## Jupyter 노트북 워크플로

`notebooks/` 에 온도 시리즈 분석 2단계 워크플로가 있습니다(실데이터 없으면 합성 데모로 바로 실행됨):

- **`01_preprocess_and_pdf.ipynb`** — 폴더(이름=온도) 불러오기 → 전처리 전/후(hot pixel·median·beam center) → PDF 변환 과정(I(q)→φ(q)→G(r)) → 온도별 `.npz` 저장
- **`02_nmf_temperature_analysis.ipynb`** — `.npz` 불러오기 → G(r) 무지개 워터폴 → **NMF**(성분 2개, 수정 가능) → 성분별 온도 기여 → NMF 분율 vs 온도 → 1st peak(1.5–1.7 Å) 가우시안 피팅으로 위치 이동 측정

노트북에 쓰이는 함수는 모두 패키지에 범용 함수로 들어 있습니다:
`clean_pattern`/`remove_hot_pixels`(전처리), `Series.from_folders`(폴더=온도 로더),
`decompose_profiles`(1D 프로파일 NMF/PCA + 분율), `fit_gaussian_peak`(가우시안 피팅),
`plot_series_waterfall`/`plot_fractions`(시각화).

## 핵심 개념

- **DataCube** — 2D(패턴 1장) / 3D(스택·시계열) / 4D(스캔) 을 동일한 인터페이스로 담는 컨테이너.
  `q_per_px`, beam center 등 캘리브레이션을 함께 들고 다닙니다.
- **마스크 규약** — Bragg/stopper 마스크는 `True`=제외, virtual detector(disk/annulus/wedge)는 `True`=포함.
- **RDF 규약** — `q = 1/d`, `G(r) = 8π ∫ q·φ(q)·sin(2π q r) dq`, Lorch/Gauss damping. series에서는 center와 scale N만 프레임별로 바뀌고 나머지 reduction 파라미터는 고정.
- **Bragg 검출** — 방위각(등방) 배경을 빼고 *strict local maximum* 만 남겨, 매끄러운 비정질 링은 잡지 않고 실제 스팟만 검출합니다.

## 테스트

```bash
pip install -e ".[dev]"
pytest -q          # 합성 데이터로 전체 파이프라인 검증 (실제 .dm4 불필요)
```

## 유의사항 (⚑)

- **q 단위**: dm4 메타데이터가 1/nm 또는 1/Å이면 자동 변환, 그 외엔 경고 후 원값 사용 — 알려진 링으로 검증하세요.
- **Scattering factor**: `abtem`의 `kirkland.json`을 쓰며, 없으면 조잡한 해석식으로 대체됩니다(peak 위치는 유효, 절대 진폭은 부정확). 정량 분석 시 실제 산란인자를 넣으세요.
- **SiOx 조성**은 가정값이므로 절대 배위수보다 **peak 위치·상대 변화**를 신뢰하세요.
