


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 

## OpenCV란

**OpenCV(Open Source Computer Vision Library)**는 컴퓨터가 사진과 영상을 보고 분석하도록 도와주는 오픈소스 라이브러리입니다.

> `사람의 눈` 역할을 대신하는 프로그램 도구입니다.
> 
> 예를 들어 사람은 신호등을 보면 빨간불인지 바로 알지만 컴퓨터는 픽셀(숫자)만 보기 때문에 OpenCV가 `픽셀을 분석`해 "빨간색이다", "차선이다", "사람이다"와 같은 `정보를 추출`합니다.

#### OpenCV가 담당하는 일

- 차선 찾기
- 신호등 색상 구분
- 도로 윤곽 검출
- 영상 전처리
- 카메라 보정

#### OpenCV 설치

```powershell
pip install opencv-python
pip install opencv-contrib-python   # 카메라기능 사용
```

#### OpenCV 이미지 구조

```powershell
import cv2

img = cv2.imread("car.jpg")
print(img.shape)
```

- OpenCV는 RGB가 아니라 BGR 순서
- shape =⇒ (480,640,3) height , widht, channel(색상BGR)

```powershell
[[[199 158 109]        # 1행1열 => 픽셀 => B,G,R
  [199 158 109]
  [199 158 109]
  ...
  [197 156 111]
  [194 153 108]
  [189 148 103]]        # 1행

 [[199 158 109]
  [199 158 109]
  [199 158 109]
  ...
  [197 156 111]
  [194 153 108]
  [190 149 104]]        # 2행

 [[199 158 109]
  [199 158 109]
  [199 158 109]
  ...
  [197 156 111]
  [194 153 108]
  [190 149 104]]

 ...

 [[ 95  88  85]
  [ 95  88  85]
  [ 95  88  85]
  ...
  [ 91  81  81]
  [ 90  80  80]
  [ 89  79  79]]

 [[ 94  86  86]
  [ 94  86  86]
  [ 95  87  87]
  ...
  [ 91  79  77]
  [ 91  79  77]
  [ 90  78  76]]

 [[ 92  84  84]
  [ 92  84  84]
  [ 92  84  84]
  ...
  [ 90  78  76]
  [ 89  77  75]
  [ 88  76  74]]]
```

BGR

- (255,255,255) —> 흰색
- (0,0,0) → 검정색

# 핵심 함수 한눈에 보기

|기능|함수|
|---|---|
|이미지 읽기|`imread()`|
|이미지 저장|`imwrite()`|
|영상 출력|`imshow()`|
|색상 변환|`cvtColor()`|
|크기 변경|`resize()`|
|자르기|NumPy 슬라이싱|
|블러|`GaussianBlur()`|
|엣지|`Canny()`|
|차선|`HoughLinesP()`|
|윤곽선|`findContours()`|
|Threshold|`threshold()`|
|형태학|`morphologyEx()`|
|카메라|`VideoCapture()`|

#### 1. 이미지 읽고 출력

```powershell
import cv2

img = cv2.imread("road.jpg")

cv2.imshow("Road", img)
cv2.waitKey(0)                  # 입력대기
cv2.destroyAllWindows()
```

#### 2. 색상변환

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

변환 종류

|변환|설명|
|---|---|
|BGR→Gray|흑백|
|BGR→HSV|색상 검출|
|BGR→RGB|Matplotlib 출력|

HSV는 신호등이나 색상 인식에 자주 사용됩니다.

#### 4. 이미지 크기 변경

```python
small = cv2.resize(img, None, fx=0.5, fy=0.5)
```

#### 5. 이미지 자르기(ROI)

ROI = Region Of Interest

```python
roi = img[100:300,200:500]
```

#### 6. 선 그리기

|함수|역할|
|---|---|
|`line()`|선|
|`rectangle()`|사각형|
|`circle()`|원|
|`putText()`|글자|

객체 인식 결과를 표시할 때 사용합니다.

#### 7. 이미지 블러(Blur)

노이즈 제거에 사용

```python
blur = cv2.GaussianBlur(img,(5,5),0)
```

종류

|함수|특징|
|---|---|
|GaussianBlur|가장 많이 사용|
|MedianBlur|소금·후추 노이즈|
|Blur|평균값|

#### 8. 엣지 검출(Canny)

Canny는 에지의 강도(Gradient)를 계산한 후 두 개의 임계값으로 분류합니다.

- 강한 에지: `max_val` 이상 → 무조건 에지로 채택
- 약한 에지: `min_val` 이상, `max_val` 미만 → 강한 에지와 연결되어 있으면 채택
- 아주 약한 에지: `min_val` 미만 → 제거

```python
edges = cv2.Canny(gray,50,150)
''' 
에지 강도   결과
   30      제거
   80      강한 에지와 연결되면 유지
  170      무조건 유지
'''
```

![[0923_7.png]]

이 과정을 거치고 나면 선만 남음

|환경|권장값 (`min`, `max`)||
|---|---|---|
|맑은 낮|`50, 150`|가장 많이 사용하는 기본값|
|흐린 날|`30, 100`||
|야간|`70, 200`||
|노이즈 많은 영상|`100, 200`||

#### 9. 허프 변환(Hough Transform)

차선을 찾을 때 사용하는 대표 기법입니다.

```python
lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi/180,
    50,
    minLineLength=50,
    maxLineGap=20
)
```

#### 10. 윤곽선 검출(Contours)

물체의 윤곽선을 찾음

활용

- 물체 개수 세기
- 면적 계산
- 중심점 찾기

#### 11. 임계값 처리(threshold)

문서 스캔이나 차선 검출에서 많이 사용

픽셀을 흑백으로 바꾸는 작업

```python
_, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
```

#### 12. 형태학 연산(Morphology)

노이즈를 제거하는 작업

|연산|역할|
|---|---|
|Erosion|작게 만들기|
|Dilation|크게 만들기|
|Opening|노이즈 제거|
|Closing|구멍 메우기|

```python
kernel = np.ones((3,3),np.uint8)
result = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
```

# 자율주행 차선 검출 전체 흐름

![[0923_2.png]]

![[0923_1.jpg]]
[Self-Driving Cars: Finding Lane Lines on the Road – AVITECH](https://images.openai.com/static-rsc-4/e6K0o40MYKnqUZ7zvGwzx1jGre-n31snROl21lLdTyREr3w5XAfxwWaxxaq1FbYs4YxHCmr5PxXzCzN248JaQOIdjmv5rmBVgaaUdZPnNv-NbwPfHSff5epbTcazNHfks1v0ji8MOpVWWCqRcVZkBwe_KC1NlBDydRI-UOUS8yI?purpose=inline)

가장 대표적인 OpenCV 프로젝트입니다.

- 원본 이미지 입력 — 도로 영상을 가져옵니다.
- Grayscale 변환 — 색상 정보를 제거해 계산량을 줄입니다.
- Gaussian Blur — 노이즈를 제거합니다.
- Canny Edge — 차선의 경계선을 찾습니다.
- ROI 설정 — 도로 영역만 남깁니다.
- Hough Transform — 직선 차선을 검출합니다.
- 원본 영상에 차선 표시 — 검출된 차선을 다시 영상에 그립니다.

**OpenCV로 차선 검출하고 CNN/YOLO 로 사람인식**

## HSV

- 신호등 인식, 물체색 구분에 강함.
- 색상 검출에 감함

```python

HSV(색조, 채도, 명도)
색조 채널은 색상의 종류를 나타내므로, 색상을 기준으로 객체를 분할해야 하는 이미지 처리 작업에 매우 유용합니다. 
채도는 무채색에서 회색 음영, 그리고 완전 포화(흰색 성분 없음)까지 다양하게 표현됩니다. 
명도 채널은 색상의 밝기 또는 강도를 나타냅니다. 
H(hue, 색상) :  
   - 색의 종류(0,360 도 각도개념)
   - OpenCV 8비트 이미지(uint8)의 표준 규격은 0~179(360도/2) 범위를 사용
S(Staturtion, 채도) : 
   - 색의 순도/진함 정도)선명도) 0~255         
   - 0: 무채색, 255:원색           
V(Value, 명도): 
   - 색의 밝기(0~255)
   - 0: 검은색, 255: 가장 밝음

HSV를 사용하는 이유
- RGB 색 공간의 색상은 세 개의 채널을 사용하여 코딩되므로 색상을 기준으로 이미지에서 객체를 분할하는 것이 더 어렵습니다.
- BGR 공간: 조명(빛)이 조금만 바뀌어도 B, G, R 세 값이 모두 크게 변하므로, 특정 색상(예: 빨간색, 파란색)을 분리해내기 어렵습니다.
- 즉, 색상의
```

H(Hue)는 색의 종류를 의미합니다. Hue가 0~179 범위를 사용합니다.

|색상|Hue(OpenCV)||
|---|---|---|
|빨강|0|빨간 신호등은 Hue가 0 근처에 있습니다.|
|노랑|약 30||
|초록|약 60||
|파랑|약 120||

![[0923_3.png]]


HSV는 원뿔처럼 표현됩니다.

- H : 어떤 색인가?
- S : 색이 얼마나 진한가?
- V : 얼마나 밝게 빛나는가?

예를 들어

|색|HSV 예시|
|---|---|
|선명한 빨강|(0,255,255)|
|어두운 빨강|(0,255,80)|
|흰색|(0,0,255)|

**HSV 사용 흐름도**

[원본 BGR 영상] (카메라 프레임)  
│  
▼ cv.cvtColor(..., cv.COLOR_BGR2HSV)  
[HSV 영상] (색상/채도/명도 분리)  
│  
▼ cv.inRange(..., 최소범위, 최대범위)  
[이진화 마스크] (찾고자 하는 색상 위치만 흰색, 나머지는 검은색)

이후 이 마스크를 활용해 `cv.bitwise_and()` 연산으로 원하는 색상의 물체만 원본 컬러로 오려내거나, 외곽선(Contour)을 따서 물체의 위치와 크기를 추적할 수 있습니다.

## HLS

- 자율주행에 많이 사용
    - 햇빛 때문에 그림자, 역광, 터널 등이 생김 → HSV 는 value가 크게 변환
    - HLS는 lightness가 상대적으로 안정적이라 흰차선, 노란차선을 구분하기 더 쉽다

### Lightness의 의미

Lightness는

> 검정과 흰색 사이의 중간 정도

를 의미합니다.

계산 개념은

L = (최대값+최소값 )/ 2

예를 들어

|RGB|Lightness|
|---|---|
|(255,255,255)|255|
|(0,0,0)|0|
|(255,0,0)|127|
|(0,255,0)|127|

빨강도, 초록도 모두 중간 밝기가 됩니다.

# HSV와 HLS 한눈에 비교

|항목|HSV|HLS|
|---|---|---|
|풀네임|Hue Saturation **Value**|Hue **Lightness** Saturation|
|밝기|Value(최대 밝기)|Lightness(**중간 밝기**)|
|색상(H)|0~179(OpenCV)|0~179(OpenCV)|
|채도(S)|0~255|0~255|
|세 번째 값|Value|Lightness|
|주 용도|**색상 검출**|**차선 검출**|
![[0923_4.png]]
![[0923_5.png]]






#### 텍스트 출력

![[0923_6.png]]

## 비디오 읽고 쓰기

[Reading and Writing Videos using OpenCV](https://opencv.org/reading-and-writing-videos-using-opencv/)
