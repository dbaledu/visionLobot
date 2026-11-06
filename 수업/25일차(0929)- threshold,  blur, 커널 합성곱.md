


## 수업목표


- **픽셀** 개념 (0 -검정)
- **채널**이란. 1채널은 흑백 이미지. 컬러 이미지는 3채널, 알파(투명도)가 있으면 4채널
    - 흑백 이미지는 각 픽셀의 밝기 값으로 표현
    - 컬러 이미지는 BGR 3채널로 3차원 배열로 표현
- 동영상 : 연속적인 프레임(하나의 이미지)
- 영상분석 파이프라인
    - 컬러이미지 → **`GrayScale`**(단순하게) → **`Blur`**(노이즈제거) → **`Threshold`**(관심영역분리:차선)
    - 블러(**Blur**) : 주변 값을 평균화
    - 사픈(Sharpen) : 중앙과 주변의 차이를 강조
    - 엣지(Edge) : 밝기가 급격하게 변화는 위치를 찾음
- **Threshold**
- 입력이미지 → 커널과 Convolution(합성곱) → 출력이미지(feature Map)
    - **커널**은 이미지에 적용할 작은 숫자 행렬
    - **합성곱**은 각 이미지와 커널의 픽셀을 각각 곱한 뒤 모두 더함
    - **픽쳐맵**은 합성곱의 결과는 이미지에서 찾아낸 특성 결과. 커널 적용 결과
    - **스트라이드** : 이동 간격
    - **패딩** : 경계처리

> **커널 = 특징을 찾는 필터**

> **Feature Map = 그 필터로 찾은 특징의 위치와 강도를 나타낸 결과**

## 실습파일
- ㅇㅇ
- 


## Threashold

- OpenCV_Threshold.pdf

> 영상의 픽셀 밝기를 기준으로 관심 영역을 분리하는 핵심 기법

#### Threashold란

- `임계값(T)`을 기준으로 픽셀을 두 영역으로 분리 → 이진화(바이너리이미지) → 0 과 255만 가짐
    - 어떠한 경계값을 기준으로 `이진화`를 시키면 물체가 뚜렷해짐
    - 복잡한 원본 → Gray 변환 → Threadhold → 객체/배경분리 : ← binary threashold
- Gray → 픽셀값 0~255 기준으로 객체와 배경을 구분
    - 0 : 검정
    - 128 : 회색 ← T 가 됨 . 128보다 작으면 검정(0) 크면 흰색(255)
        - f(x,y) > T ⇒ maxVal f(x,y) < T ⇒ minVal
    - 256 : 흰색
- `단순한 영역`으로 만드는 것
- OCR, 산업용비전, 객체검출의 `전처리`
    - 문서 : 문자와 배경 분리( 배경 흰색과 텍스트 검정)
    - 비전: 불량 영역 검사
    - OCR: 문자영역 강조
- Global Threshold
    - 고정 임계값 → 조명 변화에 취약(그림자)
- **`otsu`** 알고리즘
    - 히스토그램(정규분포곡선=가우시안) 을 분석하여 **`임계값을 자동으로 결정`**
    - 밝은 영역과 어두운 영역을 비교적 명확하게 분리될 때 유용 → 조명변화에 취약(그림자)
    - 중앙값(median) / 평균(mean)
- Adaptive Threshold
    - 이미지 전체가 아니라 **`주변 영역`**을 기준으로 임계값 결정. T가 여러 개 나올 수 있음
    - Mean / Gaussian 방식으로 지역기준값 계산
    - 지역평균 또는 가중평균에서 C를 보정
    - **`adaptiveThreshold`**함수 이용
- **`cv.threshold`**와  **`cv.adaptiveThreshold`**함수

> **AdaptiveThreshold 알고리즘에 Gaussian 함수를 적용한 경우가 조명변화에 잘 적응함.**

![[0920_1.png]]

#### Global Threshold

```python
import cv2
img = cv2.imread('book.jpg', cv2.IMREAD_GRAYSCALE)

# threshold(불러올 이미지,임계값,값이 임계값보다 클때 적용할 value, tresholding type) 
# 첫 번째는 사용된 임계값이고, 두 번째는 임계값 처리가 적용된 이미지 
ret, binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

cv2.imshow('img',img)
cv2.imshow('binary',binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### threashold type test ( cv0929_1_threshold.py )

- **`cv2.THRESH_BINARY`** : **밝은 물체를 배경에서 분리할 때** 사용(흰색 글자, 차선)
    - 이진 객체 분리. 작으면 0, 크면 255
- **`cv2.THRESH_BINARY_INV`** : 어두운 객체 분리(문서 이미지)
    - 검은 객체를 흰색으로 : Binary 반전
    - 검은 글자를 흰색 객체로 만들어 contour 등을 찾기 편합니다.
- **`cv2.THRESH_TRUNC`**
    - 밝기 상한 제한 : T 이하는 유지하고 큰 값은 T
    - **너무 밝은 영역의 밝기를 일정 수준으로 제한하고 싶을 때** 사용
    - 객체를 분리하는 목적보다는 밝기 값을 제한하는 목적
- **`cv2.THRESH_TOZERO`** : 밝은 영역만 남기기
    - 밝은 영역 강조 : 작은 값은 0
    - 밝은 부분만 남기면서 **밝기 정보도 보존하고 싶을 때** 사용
    - "어두운 것은 버리고 밝은 것은 원래 밝기 그대로 남기겠다.”
- **`cv2.THRESH_TOZERO_INV`** : 어두운 영역만 남기기
    - 어두운 영역 강조 : 큰 값은 0
    - "밝은 것은 버리고 어두운 것은 원래 값 그대로 남기겠다.”

```
Threshold type test
1. binary
2. binary_inv
3. TRUNC
4. TOZERO
5. TOZERO_INV
```
![[0920_2.png]]

![[0920_3.png]]


#### 5가지를 한 번에 비교

T = 127이라고 하겠습니다.

입력:

```
       어두움              밝음
         ↓                  ↓
입력 :   50    100    150    200
```

|Threshold|50|100|150|200|핵심|
|---|---|---|---|---|---|
|**BINARY**|0|0|255|255|밝은 객체를 흰색으로|
|**BINARY_INV**|255|255|0|0|어두운 객체를 흰색으로|
|**TRUNC**|50|100|127|127|밝기 상한 제한|
|**TOZERO**|0|0|150|200|밝은 영역만 원래 값으로|
|**TOZERO_INV**|50|100|0|0|어두운 영역만 원래 값으로|

#### Adaptive(적응형) Threshold ( cv0929_2_adaptiveThreshold.py )

적응형 임계값 설정 알고리즘은 픽셀 주변의 작은 영역을 기반으로 해당 픽셀에 대한 임계값을 결정합니다. 따라서 동일한 이미지의 여러 영역에 대해 `서로 다른 임계값`을 얻을 수 있으므로 **조명 조건이 다양한 이미지에서 더 나은 결과**를 얻을 수 있습니다.

```python
cv2.adaptiveThreshold(image,
      255,                               # 이진화 시 부여할 픽셀 최댓값(흰색)
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,   # 영역 내 픽셀에 가우시안 가중치 평균 적용
      cv2.THRESH_BINARY,                 # 일반 이진화 모드
      block_size[i],                     # 영역(블록) 사이즈  
      2)                                 # 가감 상수
```

- 조명 조건이 다른 경우 이미지의 영역별로 → 위치마다 다른 T 값 사용
- adaptive Method
    - **`cv.ADAPTIVE_THRESH_MEAN_C`**  : 임계값은 주변 영역의 평균에서 상수 **C를** 뺀 값
        - 주변 영역의 **전체적인 평균 밝기**를 보고 판단
    - **`cv.ADAPTIVE_THRESH_GAUSSIAN_C`** : 임계값은 주변 값의 가우시안 가중 합에서 상수 **C** 를 뺀 값
        - 주변 픽셀을 똑같이 보지 않고, **중심에 가까운 픽셀에 더 큰 가중치**를 주어 기준을 정한다
- 블록크기 : 인접 영역의 크기
    - 작은블록 : 지역변화에 민감. `노이즈에 민감`
        - 노이즈 제거하기 위해 필터링 전처리 필요함(blur)
    - 적절한 블록 : 보통 **11** 정도
    - 큰 블록 : 아주 작은 밝기 변화나 작은 노이즈의 영향은 줄어들고, 넓은 영역의 밝기 특성을 반영
- C : 인접 픽셀의 평균 또는 가중 합계에서 빼는 상수
    - 미세조정하여 최종 T를 계산
    - 지역평균 또는 가중평균에서 C(바이어스)를 보정
    - C가 커지면 → threshold가 낮아짐 → 기준이 느슨 → 흰색(255)으로 판정되는 픽셀이 늘어남

> **차선 검출**에서는 조명이나 그림자가 일정하지 않은 도로라면 `adaptiveThreshold`가 유용할 수 있고, 그중 `GAUSSIAN_C`는 **현재 위치 주변의 밝기 특성을 좀 더 반영하는 방식**이라고 이해

![[0920_4.png]]

![[0929_5.png]]

![[0929_6.png]]


block: 43
![[0929_7.png]]


block : 101
![[0929_8.png]]



#### otsu(오츠) Threadhold

```python
ret_otsu, th_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(f'otsu 알고리즘이 자동으로 찾은 임계값:{ret_otsu}')
imageShow('th_otsu', th_otsu)
```

![[0929_9.png]]

#### otsu와 Adaptive 알고리즘 비교

- 블록 사이즈 11인 경우

#### Morphology(형태학적) 변화

[Morphological Transformations — OpenCV Tutorials](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)

#### threshold tutorial 예제

[Basic Thresholding Operations — OpenCV Tutorials](https://docs.opencv.org/5.0/tutorials/imgproc/threshold/threshold.html#threshold-binary)

#### Canny 엣지 검출

[Canny Edge Detection — OpenCV Tutorials](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_canny/py_canny.html)

## 커널 합성곱

- OpenCV_커널-합성곱.pdf
- Kernel ⭐⭐
    - 필터 개념
    - 커널은 ‘어떤 특징을 볼 것인가’ 를 수치로 표현한다
    - 각 위치의 이미지 값과 커널 값을 곱한 뒤 모두 더한다. → 합성곱 (convolution)
    - 모두 더해 새로운 출력 픽셀을 만듬 → feature map(특징)
    - 커널 사이즈
        - 3 X 3, 5 X 5, 7 X 7
- Feature Map
    - CNN의 필터가 이미지에서 특정 특징을 찾아낸 결과를 숫자 행렬로 표현한 것
- 커널 예시
    - Blur Kernel → 주변값의 평균
    - Sobel Kernel → 방향성 밝기변화
    - Sharpen Kernel → 세부 구조 강조
        - 중앙 픽셀은 크게 반영하고 주분 픽셀을 빼서 차이를 강조

![[0920_10.png]]


- 스트라이드
    - 커널 이동 크기
    - 스트라이드(Stride)가 커진다는 것은 커널이 한 번에 이동하는 픽셀 간격이 커진다는 의미
    - 스트라이드가 작으면 촘촘하게 검사하고, 스트라이드가 크면 듬성듬성 검사
- 패딩
    - 이미지 가장자리에서는 커널 주변 픽셀이 부족함
    - 이미지 주변에 가상의 픽셀을 추가하는 방법
    - Zero / Replicate / Reflect

변화량 —> 미분 사용

#### Edge Detection

- Sobel X Kernel
- Sobel Y Kernel
- Laplacian

## Blur

- 이미지의 `잡음(noise)을 줄이고` 영상을 부드럽게 만드는 `Blur(평활화)` 방법
- Mean Blur : 주변 픽셀을 모두 동일한 비중으로 평균