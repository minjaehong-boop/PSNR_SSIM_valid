# PSNR 및 SSIM 메트릭 검증 (PSNR/SSIM Metric Validation)

이 프로젝트는 이미지 품질 메트릭인 **PSNR**과 **SSIM**의 **공식 라이브러리(`torchmetrics`) 구현**과 **커스텀 구현**이 동일한 결과를 도출하는지 검증합니다.

## 검증 목표

  * $\text{PSNR}$ 공식 구현 (`psnr.py`의 `val` 함수) vs. `torchmetrics` (`off` 함수) 일치 확인.
  * $\text{SSIM}$ 공식 구현 (`ssim.py`의 `val` 함수) vs. `torchmetrics` (`off` 함수) 일치 확인. (3x3 $\text{Box Kernel}$, $\text{reflect}$ 패딩 사용)

## 파일 목록

| 파일명 | 설명 |
| :--- | :--- |
| `psnr.py` | $\text{PSNR}$ 계산 함수 정의 |
| `ssim.py` | $\text{SSIM}$ 계산 함수 정의 (3x3 Box Kernel, Reflect Pad) |
| `draw.py` | $\text{PSNR}$과 $\text{SSIM}$을 통합 계산하고 시각화 비교 이미지를 생성하는 스크립트 |
| `gt.csv` | Ground Truth (정답) 픽셀 데이터 |
| `pr.csv` | Prediction (예측) 픽셀 데이터 |

## 사용 방법

### 1\. 검증

`gt.csv`와 `pr.csv` 파일이 준비된 상태에서, 각 스크립트를 실행하여 Official (Official) 값과 Validation (Custom) 값을 비교합니다.

```bash
# PSNR 점수 확인
python psnr.py

# SSIM 점수 확인
python ssim.py
```

### 2\. 시각적 비교 이미지 생성

```bash
# 비교 이미지 파일 생성
python draw.py
```
