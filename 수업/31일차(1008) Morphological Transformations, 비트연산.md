


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 

MCU

## **형태학적 변환(Morphological Transformations)**

- cannyEdge 이후에 추출된 윤곽선을 보정 - 끊어진 차선 연결, 노이즈 제거
- 커널: 사각형, 타원, 십자가
- 연산: 침식, 팽창, 열림, 닫힘, 탑햇, 블랙햇 …

참고사이트

[Morphology Playground](https://calebrob.com/static/morphology_playground.html)

#### Morphology 적용전
![[Morphological01.png]]

![[Morphological02.png]]

![[Morphological03.png]]

![[Morphological04.png]]

![[Morphological05.png]]


iterator
![[Morphological06.png]]


cv1008_2_test.py

모폴로지 적용 전
![[Morphological07.png]]


Morphology opening 적용 후
![[Morphological08.png]]


영상에서 차선 검출

cv1008_2_video_lane_detect_snow.py
![[Morphological09.png]]


차선이 아닌 눈 덮인 산과 들판의 경계선이 잡히는 문제

### 원인

1. **ROI의 윗변이 하늘에 걸려 있음**: `TOP_Y = 0.40`은 지평선(약 0.68)보다 훨씬 위입니다. 그 결과 ROI에 산 능선과 눈 경계가 들어옵니다.
2. **각도 필터가 너무 느슨함**: `lineFitting`의 `5*np.pi/180`은 5° 이상이면 모두 통과시킵니다. 능선, 지평선, 울타리는 기울기가 0.15~0.2(약 9~12°)라 걸러지지 않고, 이런 선들이 평균되어 X자가 됩니다.
3. **Canny를 블러 없이 4K 원본에 적용**: 아스팔트 질감과 눈 입자까지 에지로 잡힙니다.
4. **에지의 색을 보지 않음**: 차선은 흰색·노란색인데, 어떤 에지든 후보가 됩니다.

### 수정

**1) ROI를 지평선 바로 아래부터로 변경**

```python
TOP_Y = 0.70
ROI_TOP_X = (0.38, 0.58)
```

**2) Canny 전에 블러를 넣고, 흰색 부분의 에지만 남기기**

이 영상에서 차선의 밝기(HLS의 L)는 약 146~195, 아스팔트는 약 64~83이라 구분이 확실합니다.

python

```python
image_gray = convertColor(frame, cv2.COLOR_BGR2GRAY)
image_gray = cv2.GaussianBlur(image_gray, (5, 5), 0)
image_canny = cannyEdge(image_gray, 100, 200)
hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
white = cv2.inRange(hls[:, :, 1], 130, 255)        # 밝은(흰색) 픽셀
white = cv2.dilate(white, np.ones((5, 5), np.uint8))
image_canny = cv2.bitwise_and(image_canny, white)  # 차선 근처 에지만 남김
```

**3) 각도 필터를 올리기**

```python
image_lane = lineFitting(frame, lines, (0,0,255), 5, 17*np.pi/180.)   # 5° → 17°
```

17°(기울기 약 0.3)로 정한 이유는 이 영상의 실제 차선 기울기가 완만하기 때문입니다. 오른쪽 흰 실선이 약 0.38, 왼쪽 점선이 약 0.5입니다. 지난번에 쓴 0.5로 올리면 오른쪽 차선이 잘리므로, 영상마다 이 값을 확인해야 합니다.

**4) 4K 프레임 줄이기** (속도와 노이즈 모두 개선)

```python
frame = cv2.resize(frame, None, fx=0.5, fy=0.5)   # 루프에서 cap.read() 직후
```

### 참고

- 왼쪽 가장자리의 노란 실선은 기울기가 약 0.16으로 너무 완만해서 3)의 필터에 걸러집니다. 대신 내 차선의 왼쪽 경계인 점선이 잡힙니다. 노란 선까지 잡으려면 각도를 낮춰야 하는데, 그러면 능선 같은 수평선이 다시 섞이므로 균형이 필요합니다.
- 흰색 기준값 130은 낮 영상 기준입니다. 어두운 영상에서는 차선 자체가 더 어두워서(지난번 야간 이미지는 약 105~119) 값을 낮춰야 합니다. 밝기가 자주 바뀌는 영상이라면 앞서 만든 `lane_hough.py`의 백분위수 방식(`USE_BRIGHT_MASK`)이 더 안전합니다.
- 제가 테스트한 것은 올려주신 화면 캡처(약 1000px 폭)이고, `lineFitting`·`houghLinesP`·`polyROI`의 내부 구현은 보지 못했습니다. 실제 4K 프레임에서는 Canny 임계값(100/200)이나 흰색 기준값을 조금 조정해야 할 수 있습니다.
- 위 수정 후에도 엉뚱한 선이 잡히면 `lineFitting` 함수 코드를 붙여 주세요. 선분을 좌/우로 나누는 방식과 선을 그리는 범위를 확인하면 정확히 짚을 수 있습니다.

## 비트연산

- 비트 → 바이트 → KB …
- 2진수. 8진수, 16진수
- 진수변환( 2진수 ←→ 8진수)
- or, and, xor
