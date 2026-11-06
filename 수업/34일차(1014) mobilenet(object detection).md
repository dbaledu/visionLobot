
### 전날 수업 정리
- 객체 탐지 
	- 하르 캐스케이드
	- 파일 형식의 학습이 끝난 딥러닝 모델을 가져와서 사용
		- **ssd_mobilenet**
	- 인공지능 개발 프레임워크
		- 구글 colab
		- pytorch 
		- tensorflow ( 케라스 추상화)

-  **파이프라인**
1) **딥러닝 모델 로딩** 


```python
net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
```

2) **전처리: 이미지를 blob으로 변환** :  딥러닝 모델이 입력으로 사용할 수 있는 형태

```python
blob = cv2.dnn.blobFromImage(
    image,                     # 입력원본 이미지
    scalefactor=1.0,           # 픽셀값에 곱할 배율
    size=(300, 300),           # 모델이 요구하는 입력 크기 (너비, 높이)
    mean=(0, 0, 0),            # 픽셀값에서 뺄 평균값 
    swapRB=False,              # R과B 채널교체여부(BGR(OpenCV)->RGB(모델입력)로 변경)
    crop=False                 # 이미지 자르기 여부
)
# blob : 변환이 완료된 딥러닝 모델의 입력 데이터
# scalefactor=1/255 로 지정하면 0~1 사이로 정규화
# OpenCV는 일반적으로 이미지를 BGR 순서로 읽지만, 일부 딥러닝 모델은 RGB 순서
# mean=(0, 0, 0) 은 픽셀값의 중심을 이동시키는 정규화 방법. `scalefactor`와 함께 사용하면 모델에 따라 픽셀값을 -1~1 범위로 변환할 수도 있습니다.

# scalefactor=1.0 / 127.5,  mean=(127.5, 127.5, 127.5), 을 지정하면
#  변환값은 = (픽셀값 -127.5) / 127.5  ===> -1 ~ 1 사이의 값이 됨
```

##### 이미지와 blob의 차이

     원본이미지
     일반적인 OpenCV 이미지 배열: (높이, 너비, 3)
          ↓
     blobFromImage()로 전처리
      이미지 크기뿐만 아니라 색상채널, 픽셀값을 모델의 입력규격에 맞게 
      변환. 사용하는 모델에 맞춰서 전처리해야 함.   
          ↓
      딥러닝 입력 데이터 (`blob`) : (배치, 채널, 높이, 너비)
      배열 형태: (1, 3, 300, 300)  
     이미지 1장 · 채널 3개 · 높이 300 · 너비 300
     
      * 배치 : 한 번에 처리하는 이미지 수


3) **딥러닝 모델에 입력 데이터 전달 및 추론 실행**
```python
net.setInput(blob)           # 모델에 입력 데이터 전달
detections = net.forward()   # 객체 검출 실행
```
   detections 구조 확인
   - detections shape: (1, 1, N, 7)
   - 입력된 이미지 정보, 텐서플로루 SSD 차원, 검출한 객체정보, 7개의 정보
   -     0열: 배치 인덱스. 0은 첫 번째 이미지
	    1열: 클래스 ID. 모델에서 학습한 객체 ID. 1은 사람(person)
	    2열: confidence(탐지신뢰도) 점수. 약 98.66%
	    3열: startX. 객체의 경계 상자 왼쪽 좌표 xmin
	    4열: startY. 객체의 경계 상자 위쪽 좌표 ymin
	    5열: endX. 객체의 경계 상자 오른쪽 좌표 xmax
	    6열 :endY. 객체의 경계 상자 아래쪽 좌표 ymax

4) 결과 확인
```python
confidence = detections[0,0,i,2]   # 2열은 신뢰도
class_id = int(detections[0,0,i,1])  # 1열은 클래스ID
```


## 수업목표

- Object Dectection(객체 탐지)
- segmentation(분할)

## 실습파일

- cv1014_1_object_dectection_vehicle.py  : 차량 검출
- cv1014_2_object_dectection_movie.py : 동영상에서 차량 검출
- cv1014_3_lane_polygone.py : 차선영역 그리기
- cv1014_4_segmentation.py : canny, hough를 이용한 차선 영역 분할(세그멘테이션)
- cv1014_5_test.py : 테스트



## Segmentaion(분할)

- AMR 자율주행에서는 차선의 정확한 모양이나 주행 가능 영역을 알아내려면 Segmentation이 특히 유용합니다.
- GrabCut → Semantic Segmentation → Instance Segmentation(Mask R-CNN)
- 방법
	- 1) canny, hough를 이용한 검출
	- 2) ...

## 객체 검출과 분할 비교

| 구분         | Object Detection | Segmentation     |
| ---------- | ---------------- | ---------------- |
| 분석 단위      | 객체의 사각형 영역       | 픽셀 단위            |
| 결과         | 클래스, 신뢰도, 박스 좌표  | 픽셀별 클래스 또는 마스크   |
| 객체의 정확한 윤곽 | 알 수 없음           | 파악 가능            |
| 연산량        | 상대적으로 적은 편       | 모델에 따라 더 많을 수 있음 |
| 예시         | 사람을 박스로 표시       | 사람의 몸 형태대로 표시    |

>정리하면, **Object Detection은 “어디에 객체가 있는가?”를 찾고, Segmentation은 “각 픽셀이 어디에 속하는가?”** 를 찾습니다.



```python
  
from turtle import delay
from OpenCV_Utils import imageShow
from OpenCV_Functions import *
import time
  
# 차량 클래스 정의
VEHICLE_CLASS = {
    1: ('Person', (0,255,255)),   # yellow
    3: ('Car',(0,255,0)),         # Green
    4: ('Mortorbike',(255,255,0)),# cyan
    6: ('Bus',(0,164,255)),       # orange
    8: ('Truck',(0,0,255)),       # blue
    10: ('trafficLight', (255,255,0)) #
}
  
def detectVeicles(frame):
    # 이미지를 변환하여 OpenCV 딥러닝 모델이 입력 데이터 전달하고 추론 실행
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), swapRB=True)
    net.setInput(blob)
    detections = net.forward()
  
    (h,w) = frame.shape[:2]
  
    for i in np.arange(0, detections.shape[2]):
  
        ## 0.5(신뢰도)는 테스트해서 적절하게 지정
        confidence = detections[0,0,i,2]
        if confidence > CONF_THRESHOLD:  
  
            # 차량과 사람 검출
            class_id = int(detections[0,0,i,1])
            if class_id in VEHICLE_CLASS:
                name, color = VEHICLE_CLASS[class_id]
  
                # box 좌표 계산
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype('int')
                # box 그리기
                label = f'{name}: {confidence*100:.2f}%'
                cv2.rectangle(frame, (startX,startY), (endX, endY), color ,2)
                #라벨 출력
                y = startY - 15 if startY - 15 > 0 else startX + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,color,2)
  
  
def segmentLane(frame):
    '''
      허프 변환 알고리즘 기반 도로 영역 분할(Segmentation)
    '''
    TOP_Y = 0.5
    BOTTOM_Y = 1.0
  
    h, w = frame.shape[:2]
    pt1 = (w * 0.45, h*TOP_Y)
    pt2 = (w * 0.55, h*TOP_Y)
    pt3 = (w * 0.8, h*BOTTOM_Y)
    pt4 = (w * 0.2, h*BOTTOM_Y)
    roi_corners = np.array([[pt1,pt2,pt3,pt4]], dtype=np.int32)   # 2차원으로
  
    image_gray = convertColor(frame, cv2.COLOR_BGR2GRAY)
    image_edge = cannyEdge(image_gray, 50, 150)    
    image_roi = polyROI(image_edge, roi_corners)    
    lines = houghLinesP(image_roi, rho=1, theta=np.pi/180,
                        threshold=40, minLineLength=20, maxLineGap=20)
  
    if lines is None:
        return
    left_x, left_y, right_x, right_y = [], [], [], []
  
    # 선분들 중에서 대표 차선 검출
    for line in lines:
  
        x1,y1,x2,y2 = line.flatten()
        # 수직선 제외
        if(x2 == x1):
            continue
        # 수평에 가까운 노이즈 직선 제거(기울기 절대값 0.3 미만은 노이즈로 간주)
        dx = x2-x1
        dy = y2-y1
        slope = (dy)/(dx)
        if abs(slope) < 0.3:
            continue
  
        # 기울기로 좌우차선 구분
        if slope < 0 :                   # 왼쪽 차선
            left_x.extend([x1,x2])
            left_y.extend([y1,y2])
        else:                            # 오른쪽 차선 : 화면에서 오른쪽 아래로 향하는 선
            right_x.extend([x1,x2])
            right_y.extend([y1,y2])
  
    # 차선이 검출되면 y 위치의 x좌표 계산 <- 마름모꼴
    if len(left_x) > 0 and len(right_x)> 0 :
        # 1차 회귀 분석을 통한 왼쪽 차선과 오른쪽 차선의 방정식을 계산
        # y에 따른 x값을 추정하는 1차 방정식
        # 영상의 y축 최대값고 최소값을 이용하여 왼쪽 차선과 오른족 차선의 방정식을 계산
        left_poly = np.poly1d(np.polyfit(left_y, left_x, 1))   # 첫번째 인수를 주면 번째인수를 구함  (y=2x+0)
        right_poly = np.poly1d(np.polyfit(right_y, right_x, 1))  # y를 넣어서 x를 구
        # 도로 바닥부터 소질점 부근(화면 높이의 60%위치) 까지의 범위
        max_y = h
        min_y = int(h*0.6)
  
        # 좌/우 차선의 상단 및 하단 X좌표 계산
        l_x1, l_x2 = int(left_poly(max_y)), int(left_poly(min_y))
        r_x2, r_x1 = int(right_poly(min_y)), int(right_poly(max_y))
  
        # segmentation 영역을 표시하기 위한 이미지 복사
        overlay = frame.copy()
        pts = np.array([[(l_x1, max_y), (l_x2, min_y), (r_x2, min_y), (r_x1, max_y)]], dtype=np.int32)
        print(pts)
        # 영역에 색상 채우기
        cv2.fillPoly(overlay, pts, (0,0,255))
        # 원본 이미지와 합성
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
  
 
#---------------------------------------
# 메인 프로세스
#---------------------------------------    
CONF_THRESHOLD = 0.2
  
# 딥러닝 모델 로딩
pb_path = 'model/frozen_inference_graph.pb'  # tensorflow 모델 가중치 파일.# TensorFlow 동결 추론(inference) 그래프 모델
pbtxt_path = 'model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'  # 모델 구조 정의 파일
net = cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)
  
# 비디오 로딩
videoPath = './images/movie/city_driving_10s.mp4'  #고속도로4-hd_1280_720_25fps.mp4' # '12127133_2160_3840_30fps.mp4'  12127133_2160_3840_30fps.mp4
cap = cv2.VideoCapture(videoPath)
if not cap.isOpened():
    print('비디오 파일을 열 수 없습니다.')
    exit()
  
windowName = 'DNN VEhicle Detection(q/ESC:exit)'
cv2.namedWindow(windowName, cv2.WINDOW_GUI_EXPANDED)
  
# 비디오의 프레임 레이트와 delay 계산
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps) if fps > 0 else 30
  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('비디오 재생 완료')
        break
  
    detectVeicles(frame)
    segmentLane(frame)
  
    cv2.imshow(windowName, frame)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q') or key == 27:  # q ESC
        break
  
# 리소스 해제
cap.release()
cv2.destroyAllWindows()
```

##### 이미지 좌표계부터 이해하기
![[이미지좌표계.png]]

OpenCV에서는 다음과 같습니다.
- `x`가 증가하면 화면에서 오른쪽으로 이동합니다.
- `y`가 증가하면 화면에서 아래쪽으로 이동합니다.
	- 오른쪽 차선 : 화면에서 왼쪽 기준에서 오른쪽 아래로 향하는 직선
