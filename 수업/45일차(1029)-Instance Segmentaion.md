


## 수업목표
- Instance Segmentation - 교재 4장 참조
- Mask R-CNN 모델을 이용하여 분할
	- **Mask R-CNN 알고리즘을 활용한 도로 위 객체(인스턴스)의 검출 및 세그멘테이션(분할)**
- YOLO 모델을 이용하여 분할

## 실습파일
- cv1029_1_Instance_Segmentation_RCN.ipynb
- cv1029_2_instance_segmentation_YOLO.ipynb
- 

## 참고사이트
- [pytorch pre-tained model](https://docs.pytorch.org/vision/stable/models.html?utm_source=chatgpt.com)

#### Instance Segmentation과 Semantic Segmentation 차이

- 같은 종류의 객체를 각각 구분하느냐가 핵심 차이
- Object Detection(객체 검출) 은 **바운딩 박스와 클래스** 등을 출력하고, 두 세그멘테이션 기법은 **객체나 영역의 픽셀 경계**까지 구분한다는 차이가 있습니다.

|         | Semantic              | Instance                     |
| ------- | --------------------- | ---------------------------- |
|         | 의미론적 분할               | 개체별 분할                       |
|         | 같은 종류는 **같은 클래스**     | 같은 종류도 **개체별로 구분**           |
| 출력 결과   | 클래스별 픽셀 마스크           | 객체별 픽셀 마스크                   |
| 객체 개수파악 | 마스크만으로는 어려움           | 가능                           |
| 대표 활용   | 주행 가능 도로, 차선 주변 영역    | 앞차 추적, 개별 보행자 추적             |
| 소스구현    | 이미지의 각 픽셀에 클래스 번호를 부여 | 객체마다 별도의 마스크, 클래스, 신뢰도 등을 반환 |

가중치 클래스 목록

| 인덱스 | 영문 클래스           | 한글 설명       |
| --- | ---------------- | ----------- |
| 0   | `__background__` | 배경          |
| 1   | `person`         | 사람          |
| 2   | `bicycle`        | 자전거         |
| 3   | `car`            | 자동차         |
| 4   | `motorcycle`     | 오토바이        |
| 5   | `airplane`       | 비행기         |
| 6   | `bus`            | 버스          |
| 7   | `train`          | 기차          |
| 8   | `truck`          | 트럭          |
| 9   | `boat`           | 보트          |
| 10  | `traffic light`  | 신호등         |
| 11  | `fire hydrant`   | 소화전         |
| 12  | `N/A`            | 사용하지 않는 인덱스 |
| 13  | `stop sign`      | 정지 표지판      |
| 14  | `parking meter`  | 주차 요금 계량기   |
| 15  | `bench`          | 벤치          |
| 16  | `bird`           | 새           |
| 17  | `cat`            | 고양이         |
| 18  | `dog`            | 개           |
| 19  | `horse`          | 말           |
| 20  | `sheep`          | 양           |
| 21  | `cow`            | 소           |
| 22  | `elephant`       | 코끼리         |
| 23  | `bear`           | 곰           |
| 24  | `zebra`          | 얼룩말         |
| 25  | `giraffe`        | 기린          |
| 26  | `N/A`            | 사용하지 않는 인덱스 |
| 27  | `backpack`       | 배낭          |
| 28  | `umbrella`       | 우산          |
| 29  | `N/A`            | 사용하지 않는 인덱스 |
| 30  | `N/A`            | 사용하지 않는 인덱스 |
| 31  | `handbag`        | 핸드백         |
| 32  | `tie`            | 넥타이         |
| 33  | `suitcase`       | 여행 가방       |
| 34  | `frisbee`        | 플라잉 디스크     |
| 35  | `skis`           | 스키          |
| 36  | `snowboard`      | 스노보드        |
| 37  | `sports ball`    | 스포츠공        |
| 38  | `kite`           | 연           |
| 39  | `baseball bat`   | 야구 배트       |
| 40  | `baseball glove` | 야구 글러브      |
| 41  | `skateboard`     | 스케이트보드      |
| 42  | `surfboard`      | 서핑보드        |
| 43  | `tennis racket`  | 테니스 라켓      |
| 44  | `bottle`         | 병           |
| 45  | `N/A`            | 사용하지 않는 인덱스 |
| 46  | `wine glass`     | 와인잔         |
| 47  | `cup`            | 컵           |
| 48  | `fork`           | 포크          |
| 49  | `knife`          | 칼           |
| 50  | `spoon`          | 숟가락         |
| 51  | `bowl`           | 그릇          |
| 52  | `banana`         | 바나나         |
| 53  | `apple`          | 사과          |
| 54  | `sandwich`       | 샌드위치        |
| 55  | `orange`         | 오렌지         |
| 56  | `broccoli`       | 브로콜리        |
| 57  | `carrot`         | 당근          |
| 58  | `hot dog`        | 핫도그         |
| 59  | `pizza`          | 피자          |
| 60  | `donut`          | 도넛          |
| 61  | `cake`           | 케이크         |
| 62  | `chair`          | 의자          |
| 63  | `couch`          | 소파          |
| 64  | `potted plant`   | 화분          |
| 65  | `bed`            | 침대          |
| 66  | `N/A`            | 사용하지 않는 인덱스 |
| 67  | `dining table`   | 식탁          |
| 68  | `N/A`            | 사용하지 않는 인덱스 |
| 69  | `N/A`            | 사용하지 않는 인덱스 |
| 70  | `toilet`         | 변기          |
| 71  | `N/A`            | 사용하지 않는 인덱스 |
| 72  | `tv`             | TV          |
| 73  | `laptop`         | 노트북         |
| 74  | `mouse`          | 마우스         |
| 75  | `remote`         | 리모컨         |
| 76  | `keyboard`       | 키보드         |
| 77  | `cell phone`     | 휴대전화        |
| 78  | `microwave`      | 전자레인지       |
| 79  | `oven`           | 오븐          |
| 80  | `toaster`        | 토스터         |
| 81  | `sink`           | 싱크대         |
| 82  | `refrigerator`   | 냉장고         |
| 83  | `N/A`            | 사용하지 않는 인덱스 |
| 84  | `book`           | 책           |
| 85  | `clock`          | 시계          |
| 86  | `vase`           | 꽃병          |
| 87  | `scissors`       | 가위          |
| 88  | `teddy bear`     | 곰 인형        |
| 89  | `hair drier`     | 헤어드라이어      |
| 90  | `toothbrush`     | 칫솔          |

#### 모델 예측 결과

##### 각 항목의 의미

|키|의미|저장된 데이터|용도|
|---|---|---|---|
|`boxes`|객체 위치|바운딩 박스 좌표 4개|객체가 있는 위치 표시|
|`labels`|객체 종류|클래스 인덱스|자동차, 사람 등 분류|
|`scores`|검출 신뢰도|0~1 범위의 점수|검출 결과 필터링|
|`masks`|객체 영역|픽셀별 마스크 값|객체의 실제 모양 분할|

#### cv1029_1_Instance_Segmentation_RCN.ipynb

##### 전체소스
```python
#----------------------------------------------
# Mask R-CNN 알고리즘을 활용하여 도로 위 객체(인스턴스)를 감지하고 각 객체의
# 픽셀 단위 마스크를 추출하여 처리하는 클래스
#----------------------------------------------

class RoadInstanceSegmenter:
    def __init__(self, device=None):
        # 사용할 연산장치 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # COCO 데이터셋으로 사전 학습된 기본 가중치
        self.weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  # DEFAULT => COCO_V1
        # Mask R-CNN 모델 객체 생성
        self.model = maskrcnn_resnet50_fpn(weights = self.weights)
        # 추론 모드(Dropout, 배치정규화 비활성화)
        self.model.eval()
        # 이미지 전처리
        self.transforms = self.weights.transforms()
        # COCO 데이터셋 category(클래스명 목록) 정의
        self.categories = self.weights.meta['categories']    # List
        # print( {i: name for i, name in enumerate(self.categories)})  # Dic 구조 출력

    #-----------------------------------------------
    # 입력 이미지에 대한 객체 및 마스크 예측
    #-----------------------------------------------
    def predict(self, Image, threshold=0.5):   # 보수적으로 한다면 0.7로 지정
        # 이미지를 텐서로 변환
        input_tensor = self.transforms(Image).unsqueeze(0).to(self.device)

        # 기울기 계산 비활성화
        with torch.no_grad():
            prediction = self.model(input_tensor)[0]

        # 예측 결과값을 numpy array로 변환
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()

        # 신뢰도 점수가 threshold 값보다 큰 객체들만 선택
        keep = scores > threshold
        return {'masks': masks[keep],
                'labels': labels[keep],
                'scores': scores[keep],
                'boxes': boxes[keep]}

    #-----------------------------------------------
    # 검출된 객체별로 랜덤하게 색상을 처리하여 시각화
    #-----------------------------------------------
    def visualize(self, image, prediction, alpha=0.5):
        # 연산을 위해 numpy array 변환
        image_np = np.array(image).copy()
        h, w, c = image_np.shape

        # 시각화에 사용할 정보
        masks = prediction['masks']
        labels = prediction['labels']
   
        # 객체마다 서로 다른 색상을 지정
        np.random.seed(42)  # 랜덤한 색상 고정 및 랜덤 색상 추출
        # size=(len(masks), 3)  <- 객체수만큼 색상 3개 [R,G,B]  지정
        colors = np.random.randint(0, 255, size=(len(masks),3), dtype=np.uint8)

        # 마스크를 씌울 이미지 복사
        blended = image_np.copy()

        for i, (mask, label) in enumerate(zip(masks, labels)):
            # 임계값 보다 큰 객체만 처리
            m = mask[0] > 0.5
            color = colors[i]
            # 원본 이미지에 랜덤 색상을 알파값을 적용하여 덧씌움
            blended[m] = (image_np[m] * (1 - alpha) + color * alpha).astype(np.uint8)


        plt.figure(figsize=(10, 10))
        # 원본 이미지 출력
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        # Mask R-CNN 마스크 적용 이미지
        plt.subplot(1, 2, 2)
        plt.imshow(blended)
        plt.title('Mask R-CNN Instnace Segmentation')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #image_path = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
    #image_path = '/content/drive/MyDrive/Colab Notebooks/dataset/images/person1.jpg'
    #image_path = 'https://cdn.pixbay.com/photo/'
    image_path='/content/drive/MyDrive/Colab Notebooks/dataset/images/pedestrians.jpg'
    origin_image = load_image(image_path)

    # RoadInstanceSegmenter 클래스 인스턴스 생성
    insSeg = RoadInstanceSegmenter()

    # Mask R-CNN 세그멘테이션 추론
    prediction = insSeg.predict(origin_image, threshold=0.7)

    # Mask R-CNN 시각화
    insSeg.visualize(origin_image, prediction)
```

##### 파이프라인

프로그램은 크게 **[이미지 로드] ➡️ [모델 준비] ➡️ [예측 및 추론] ➡️ [시각화]**의 4단계 파이프라인으로 구성되어 있습니다.

#### **Step 1: 이미지 로드 및 전처리 (`load_image`)**

- 입력 소스가 로컬 파일 경로인지, 웹 URL(http/https)인지 판별합니다.
- 웹 URL인 경우, 차단을 방지하기 위해 `User-Agent` 헤더를 설정하여 HTTP 요청을 보낸 뒤 바이트 스트림을 이미지로 변환합니다.
- 모든 이미지는 PIL을 거쳐 RGB 형식의 일관된 데이터로 포맷팅됩니다.

#### **Step 2: 세그멘테이션 클래스 정의 및 모델 로드 (`RoadInstanceSegmenter`)**

- GPU(`cuda`) 사용이 가능한지 확인하고 연산 장치를 설정합니다.
- COCO 데이터셋으로 사전 학습된 가중치(`MaskRCNN_ResNet50_FPN_Weights.DEFAULT`)와 이에 맞는 이미지 변환 툴(`transforms`)을 불러옵니다.
- `maskrcnn_resnet50_fpn` 모델을 생성하고 평가 모드(`model.eval()`)로 전환합니다.

#### **Step 3: 객체 검출 및 세그멘테이션 추론 (`predict`)**

- 불러온 이미지를 텐서(Tensor) 구조로 변환하여 모델에 입력합니다.
- 모델 추론 결과로부터 객체의 위치 정보(`boxes`), 클래스 레이블(`labels`), 신뢰도 점수(`scores`), 픽셀 마스크 정보(`masks`)를 추출합니다.
- 설정한 임계값(Threshold, 예: 0.7)보다 높은 신뢰도를 가진 객체 데이터만 필터링하여 반환합니다.

#### **Step 4: 결과 시각화 (`visualize`)**

- 검출된 객체마다 서로 다른 색상을 랜덤하게 부여합니다.
- 원본 이미지 위에 추출된 마스크 영역을 투명도(`alpha`)를 적용하여 덧씌웁니다(Blending).
- `matplotlib`를 사용하여 **원본 이미지(Original Image)**와 **마스크가 적용된 세그멘테이션 이미지**를 좌우로 나란히 시각화하여 한눈에 비교할 수 있도록 출력합니다.


## YOLOv8n-seg

#### result output

| Attribute           | Type            | Shape         | Description                         |
| ------------------- | --------------- | ------------- | ----------------------------------- |
| `result.masks`      | `Masks`         | `(N)`         | Instance masks.                     |
| `result.masks.data` | `torch.uint8`   | `(N,H,W)`     | Binary masks, values `0` or `1`.    |
| `result.masks.xy`   | `np.float32`    | `list[(P,2)]` | Pixel polygons.                     |
| `result.masks.xyn`  | `np.float32`    | `list[(P,2)]` | Normalized polygons.                |
| `result.boxes`      | `Boxes`         | `(N)`         | Instance boxes/classes/confidences. |
| `result.boxes.cls`  | `torch.float32` | `(N,)`        | Class IDs; cast to `int` for names. |

#### cv1029_2_instance_segmentation_YOLO.ipynb

##### 전체소스

```python
import cv2
import time
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image


import requests
import io
#-------------------------------------------
# 로컬 파일 또는 웹 URL에서 이미지 가져옴
#-------------------------------------------
def load_image(source:str) -> Image.Image:
    '''
    로컬파일 경로 또는 웹 URL 이미지를 입력받아 PIL 이미지 객체로 변환
    '''
    # 웹 URL로 시작하는지 검사
    if source.startswith('http://') or source.startswith('https://'):
        # 서버에 잘못된 접근을 방지하기 위해 헤더 설정
        # User-Agent를 지정하지 않으면 Python requests 라이브러리의 기본 User-Agent 등 전달
        # 서버의 정책에 따라 응답이 차단 또는 오류 응답이 올 수 있음
        headers = {"User-Agent":'Mozilla/5.0'}
        response = requests.get(source, headers=headers, timeout=10)
        # HTTP 요청 결과에 오류가 있으면 예외를 발생(400번대와 500번대 상태 코드)
        response.raise_for_status()
        # 바이트 스트림 데이터를 이미지 데이터로 변환 -> RGB 변환
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    else:
        #로컬 환경의 파일에서 이미지 불러옴
        image = Image.open(source).convert('RGB')
    return image


class YOLOInstanceSegmenter:
    # yolov8n-seg 모델 로드
    def __init__(self, model_name='yolov8n-seg.pt'):
        print('YOLO 세그멘테이션 모델 로딩중...', )
        self.model = YOLO(model_name)

    # 이미지 내에서 객체의 위치 및 픽셀 단위의 마스크 영역 예측
    def predict(self, image, conf=0.25):
        result = self.model(image, conf=conf)
        return result[0]

    # 검출된 객체의 윤각 마스크와 레이블을 이미지에 그려서 시각화 처리
    def visualize(self, image, result):
        '''
        검출된 객체의 윤각 마스크와 레이블을 이미지에 그려서 시각화 처리

        Args
          image : 원본이미지
          result : Predicts 객체(예측결과)
        '''
        # 세그멘테이션 Mask가 씌워진 최종 결과 이미지 생성
        annotated_img = result.plot(())
        # BGR -> RGB 변환
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.title('YOLO Instance segmentation')
        plt.axis('off')
        plt.imshow(annotated_img_rgb)
        plt.show()


if __name__ == '__main__':
    # 이미지 경로
    image_path = "https://ultralytics.com/images/bus.jpg"
    image = load_image(image_path)

    # YOLOInstancesegmneter 객체 생성
    # -n(nano): 가장 가볍고 속도가 바른 경량 모델
    # COCO 데이터셋으로 사전 학습
    # yolo8n.pt: 객체 탐지 전용
    # yolo8n-seg.pt: 인스턴스 세그멘테이션 전용 
    # yolo8n-sem.pt: 시멘틱 세그멘테이션 전용 
    # yolo8n-cls.pt: Classification 전용
    # yolo8n-pose.pt: 포즈(인체 자세 추측) 전용 - 인체관절 17개
    seg = YOLOInstanceSegmenter('yolov8n-seg.pt')

    # 모델 예측
    result = seg.predict(image_path)

    # 결과 시각화
    seg.visualize(image, result)

```


##### 파이프라인

전체적인 코드 실행 흐름은 다음과 같은 단계로 구성되어 있습니다.

```
[단계 1: 환경 준비] ──> [단계 2: 이미지 로드] ──> [단계 3: 모델 로드 및 예측] ──> [단계 4: 시각화]
```

#### **Step 1: 패키지 설치 및 라이브러리 임포트**

- YOLO 모델 구동을 위해 `ultralytics` 패키지를 설치합니다.
- 이미지 처리 및 시각화를 위해 `OpenCV(cv2)`, `PIL(Image)`, `matplotlib.pyplot` 및 웹 통신을 위한 `requests` 라이브러리를 임포트합니다.

#### **Step 2: 이미지 로드 (`load_image` 함수)**

- 입력된 `source` 주소가 웹 URL(`http://` 또는 `https://`)인지 로컬 파일 경로인지 판별합니다.
- 웹 이미지인 경우 `requests`를 통해 데이터를 가져와 바이트 스트림으로 읽은 뒤, PIL 이미지 객체(RGB 형식)로 변환하여 반환합니다.

#### **Step 3: YOLO 클래스 정의 및 예측 (`predict`)**

- **세그멘테이션용(`YOLOInstanceSegmenter`)** 및 **디텍션용(`YOLOObjectDetector`)** 클래스가 정의되어 있습니다.
- 클래스 초기화 시 지정된 가중치 파일(`.pt`)을 로드하고, `predict` 메서드를 통해 입력 이미지에 대한 예측을 수행하여 클래스, 신뢰도(Confidence), 바운딩 박스 정보 등이 담긴 `Results` 객체를 반환합니다.

#### **Step 4: 결과 시각화 (`visualize`)**

- 모델의 예측 결과(`result`)에 포함된 `.plot()` 메서드를 사용하여 원본 이미지 위에 감지된 객체의 박스, 마스크, 레이블을 그립니다.
- 그 후, BGR 채널 포맷을 Matplotlib에서 사용할 수 있도록 RGB로 변환하여 화면에 출력합니다.