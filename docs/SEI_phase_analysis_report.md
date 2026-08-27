# 4D-STEM 기반 비정질 Li-SEI 상(相) 분석 — 요약 보고서

> 시료: 비정질 Li-SEI (P-Cu), dm4 STEM SI (survey + scan + CBED/NBD)
> 후보 물질: LiF, Li2O, Li3N, Li2CO3, Li2S
> 분석 노트북: `notebooks/06_fcstem_cepstral.ipynb` (nb6), 보조 `notebooks/05_nbed_pca_sf_rdf.ipynb`
> 산출물: dm4와 같은 폴더의 `nb6_outputs/` (그림 PNG + CSV)

이 문서는 지금까지의 분석을 **① 배경 → ② 방법 → ③ 결론 → ④ 한계** 4블록으로 정리한 것이다.
각 절 끝에 대응하는 노트북 셀(§)과 저장 파일을 표기했으므로, 숫자는 항상 해당 CSV에서 확인한다.

---

## ① 배경 (Background)

- **목적**: 비정질 Li-SEI 안에 어떤 Li 화합물(LiF/Li2O/Li3N/Li2CO3/Li2S)이, 어디에, 결정질인지 비정질인지
  구분해 규명한다. NBD(nanobeam diffraction) 4D-STEM 데이터의 링·피크만으로 어디까지 말할 수 있는지가 핵심 질문.
- **가용 정보**
  - EDS: **O, C가 많고** 그 다음 **N, F, S** — 5개 후보 모두 화학적으로 가능.
  - **EELS 없음** (측정 곤란) → 회절(링/피크)과 실공간 virtual image로만 판단해야 함.
  - **기판에 Cu 없음** → Cu/Cu2O/CuO 후보는 기본 제외 (`INCLUDE_SUBSTRATE=False`).
- **데이터 성격**: 비정질 halo(약하고 넓음) + 곳곳의 작은 결정 알갱이(다결정 Bragg 스팟)가 섞여 있음.
  그래서 "평균 패턴"만 보면 결정 신호가 희석된다 → MAX 프로젝션·스팟 검출을 병행한다.

---

## ② 방법 (Methods)

### 2.1 캘리브레이션 (검증 완료)
- dm 메타데이터 = **0.043888 1/nm/px**. 변환 체인: `÷10` = 0.0043888 1/Å/px `× DET_BIN(2)` = **0.008778 1/Å/px**.
- 주의: `Q_PER_PX`(config)는 **합성데이터 전용 fallback**이며, 실데이터는 메타데이터를 자동 사용한다.
  `0.043888`을 1/Å 자리에 넣으면 안 됨(1/nm ≠ 1/Å, 거리 10배 오차).
- q = 1/d 관례(결정학적, Å⁻¹) 사용.
- 셀 §2c · 파일 `02c_calibration_center.png`

### 2.2 배경 제거·중심빔 제거 (사용자 검증 항목)
- 직접빔 때문에 I(q)가 급강하 → 단순 argmax는 빔 어깨를 잡는다. `find_fsdp`는 **log 공간에서 빔 배경을
  반복 적합**해 빼고, **residual의 내부 봉우리**만 링으로 인정한다(빔 꼬리·창 경계를 링으로 오인하지 않음).
- **검증 지표**(§2n): residual에서 `q < q_beam`(빔 영역) 크기 ÷ 링 봉우리 크기.
  이 값이 **< 0.5**이면 중심빔이 baseline에 흡수되어 제거된 것. 물질 평균 I(q)와 MAX I(q) 둘 다 점검.
- → **결론적으로 배경·중심빔 제거는 적절**하며, residual 봉우리 위치(=링 위치)를 신뢰할 수 있다.
- 셀 §2n · 파일 `02n_background_beam_verify.png`

### 2.3 비정질 첫 링(FSDP) 추출
- 산란 강한 상위 `STRONG_FRAC`(20%) 위치만 골라 **빔중심 정렬 후 평균** → 약한 FSDP를 증폭.
- 셀 §2d · 파일 `02d_strong_signal_fsdp.png`

### 2.4 결정질(다결정) 링 분석 — 두 방식
- **(a) MAX 프로젝션 radial**: 검출기 픽셀별 스캔 전체 최댓값 → 곳곳의 Bragg 스팟을 모아 powder-like 링.
  셀 §2g · 파일 `02g_max_crystalline.png`, `02g_max_rings.csv`
- **(b) 스팟-반경 히스토그램**: 개별 Bragg 스팟을 검출→ 각 스팟 반경(|q|=1/d)을 히스토그램.
  radial 적분이 스팟을 빈 각도와 평균해 희석하는 문제를 피함. **한 상당 ≥2개 링 일치**를 존재 신뢰 기준으로 둠.
  셀 §2j · 파일 `02j_spot_histogram.png`, `02j_spot_rings.csv`

### 2.5 상(相) 매칭
- 측정 링 q를 각 화합물의 d-spacing 링(`COMPOUND_RINGS`)과 Gaussian-위치가중으로 매칭(`match_rings`).
- 셀 §2e/§2f · 파일 `02e_ring_match.png`, `02f_compound_graphs.png`, `02f_ring_map.png`

### 2.6 신뢰성 검정 (핵심)
- **Null model**: 실제 화합물 매칭 점수 vs 무작위 d-spacing "가짜 상"의 점수 분포를 비교.
- **Bonferroni 보정**(5개 상 다중비교, 임계 0.01) + 링 고유성(어느 상으로도 설명 안 되는 링) 점검.
- 셀 §2k

### 2.7 실공간 virtual image (두 신규 작업)
- **[TASK 1] 링 기반**(§2o): 다결정 powder 링 하나하나를 **환형(링 전체)** 검출기로 잡아 virtual image.
  단 원시 DF는 두께에 지배되므로 **총 산란으로 나눠(두께 정규화)** `structural_map`=링DF/총산란 →
  '그 링 성분의 공간 편중'만 남김. 각 링에 d-spacing이 맞는 예상 상 라벨.
  파일 `02o_ring_virtual_images.png`, `02o_ring_phases.csv`, `02o_ring_maps.csv`
- **[TASK 2] 대표 피크 기반**(§2p): 링패턴 속 **대표적으로 밝은 스팟**을 골라 **2D 가우시안으로 sub-pixel
  정밀 피팅** → 그 중심에 ~2σ 조리개로 virtual image → 그 반사를 내는 결정 알갱이의 실공간 위치.
  파일 `02p_gaussian_peak_df.png`, `02p_gaussian_peak.csv`, `02p_peak_virtual_image.npy`
- (참고) 화합물 링별 두께정규화 조성 지도 §2l `02l_composition_maps.png`; 단일 스팟 DF §2m `02m_*`.

### 2.8 원자간 거리·상 분리 (보조)
- **EWPC/켑스트럼**(log→역FFT): 배경 제거 없이 원자간 거리 신호. 셀 §3
- **FC-STEM 거리 밴드별 fluctuation 이미지**: 영역별 구조 차이 매핑. 셀 §4
- **켑스트럼 프로파일 NMF(k)**: 상 분리 → 영역별 회절 링 비교 §6e (`06e_region_rings.png`).
- 후보 지문 비교/ supervised 언믹싱: §6b/§6c/§6d.

---

## ③ 결론 (Results)

> 정확한 수치는 각 CSV에서 읽는다. 아래는 이번 데이터에서 얻은 대표 결과다.

1. **결정질과 비정질이 함께 존재**한다. 평균 halo(비정질) 위에, MAX/스팟 히스토그램에서 여러 개의
   날카로운 다결정 링이 나온다 → 국소 결정 알갱이가 산재.
2. **공간적으로 불균일**하다(가장 견고한 결과). §4 FC-STEM 밴드 이미지·§5 NMF·§6e 영역별 링 비교에서
   **영역마다 구조가 다르다**(서로 다른 물질/상이 공존). TASK 1·2의 virtual image도 특정 링/피크 성분이
   **실공간 특정 영역에 편중**됨을 보여준다.
3. **측정된 링 d-spacing**(대표값, 정확치는 `02g_max_rings.csv`/`02j_spot_rings.csv`):
   결정 링 약 **d ≈ 4.2, 3.1, 2.1, 1.5 Å** 부근 + 스팟 히스토그램에서 다수의 powder 링.
4. **원소**(EDS): C, O, N, F, S 존재 → 5개 후보 모두 화학적으로 가능.
5. **TASK 1/2 virtual image**: 각 링·대표 피크가 실공간 어디에 있는지 성공적으로 국소화. 라벨은 그 d에
   맞는 "후보 상"을 표시(확정 아님, ④ 참조).

---

## ④ 한계 — 무엇을 주장할 수 있고 없는가 (Limits)

이 부분이 가장 중요하다. **회절(링/피크)만으로 특정 Li 상을 유일하게 이름 붙이는 것은 신뢰할 수 없다.**

- **신뢰성 검정 결과**(§2k): 5개 후보 중 **LiF만 p ≈ 0.046으로 약하게** 유의, 그러나 **Bonferroni 보정
  (5중 비교)을 통과하지 못함**. Li2CO3/Li3N/Li2O/Li2S는 무작위 수준(p ≈ 0.32–0.73).
- **원인**: Li-경원소 화합물들의 d-spacing이 **심하게 겹친다**(예: LiF/Li2O/Li3N의 최근접이웃 ~2.0 Å).
  낮은 q_max에서 shell이 넓어져(Δr ≈ 1/q_max) 참조들이 서로 collinear → 매칭이 임의로 갈린다.
  또한 어느 상으로도 깔끔히 설명되지 않는 링(예 d ≈ 4.7, 1.5, 1.3, 1.15 Å 부근)이 남는다.
- **약한 신호**: 비정질 halo가 약해 FSDP 신뢰도(conf)가 낮은 구간이 있음 → 절대 캘리브레이션 강제(`CALIB_R_TARGET`)는
  기본 사용하지 않고 메타데이터를 신뢰.

**말할 수 있는 것 ✅**
- 결정질 + 비정질이 공존한다.
- 영역마다 구조가 다르다(공간 이질성) — 서로 다른 상이 섞여 있다.
- C, O, N, F, S가 존재한다(EDS).
- 특정 링/피크 성분이 실공간 어느 영역에 있는지(TASK 1·2).

**말할 수 없는 것 ❌**
- "이 링 = 확정적으로 이 Li 화합물" — 회절만으로는 유일 지정 불가(d-spacing 중첩 + 약신호).
- 정량 상 조성(§6d NNLS 언믹싱은 참조 축퇴로 신뢰 낮아 참고용).

**정리**: 회절/virtual image는 *어디에 무엇이 있는지의 후보와 공간 분포*까지는 강하게 보여주지만,
*상의 최종 확정*에는 EELS(Li-K, 각 화합물 fine structure)가 필요하다. 현재 EELS가 없으므로,
본 분석은 "후보 + 공간 분포 + 결정/비정질 구분"까지를 신뢰 구간으로 제시한다.

---

## 부록 A — 재현 방법 (섹션 → 셀 → 파일)

| 블록 | 노트북 셀 | 저장 파일 |
|---|---|---|
| 캘리브레이션 | §2c | `02c_calibration_center.png` |
| 배경·중심빔 검증 | §2n | `02n_background_beam_verify.png` |
| 비정질 FSDP | §2d | `02d_strong_signal_fsdp.png` |
| MAX 결정 링 | §2g | `02g_max_crystalline.png`, `02g_max_rings.csv` |
| 스팟 히스토그램 | §2j | `02j_spot_histogram.png`, `02j_spot_rings.csv` |
| 상 매칭 | §2e/§2f | `02e_ring_match.png`, `02f_*` |
| **신뢰성 검정** | §2k | (콘솔 출력: null model + Bonferroni) |
| 두께정규화 조성 | §2l | `02l_composition_maps.png`, `02l_composition.csv` |
| **[TASK 1] 링 virtual image** | §2o | `02o_ring_virtual_images.png`, `02o_ring_phases.csv`, `02o_ring_maps.csv` |
| **[TASK 2] 가우시안 피크 virtual image** | §2p | `02p_gaussian_peak_df.png`, `02p_gaussian_peak.csv` |
| 단일 스팟 DF | §2m | `02m_single_spot_df.png`, `02m_spot_overview.png` |
| EWPC/켑스트럼 | §3 | (프로파일) |
| FC-STEM 밴드 | §4 | (fluctuation 이미지) |
| NMF 상 분리 | §5 | (성분/지도) |
| 영역별 링 비교 | §6e | `06e_region_rings.png` |
| 지문/언믹싱 | §6b/§6c/§6d | `06b_*`, `06c_*`, `06d_compound_unmix.png` |

실행: nb6를 위→아래로 실행. `DM4_PATH`가 존재하면 실데이터, 없으면 합성데이터로 자동 동작.
출력은 dm4 폴더의 `nb6_outputs/`에 PNG+CSV로 저장된다.

## 부록 B — 참고 문헌
- Pidaparthy & Zuo, *Ultramicroscopy* 248 (2023) 113718 — FC-STEM, 링(RINGS) 기반 상 확인 방법.
