

## 수업목표
-  Semantic Segmentation
	- **사전 학습된 딥러닝 모델(DeepLabV3-ResNet101)**을 활용하여 도로 이미지 내에서 차량(자동차, 버스, 오토바이, 자전거) 및 보행자(사람) 등의 객체를 픽셀 단위로 정밀하게 분할(Semantic Segmentation)하고 시각화
- Segmentagion Evaluate : 
	- **시맨틱 세그멘테이션(Semantic Segmentation)**을 수행하고, 이에 대한 **평가 지표를 계산 및 시각화**하는 엔드투엔드(End-to-End) 검증 도구

## 실습파일
- cv1028_1_segmentation.ipynb
- cv_1028_2_Segmentagion Evaluate.ipynb


### 사전 학습된 모델 사용

```python
# 방식1
self.weights = DeepLabV3_ResNet101_Weights.DEFAULT
self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)

# 방식2
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
result = model(input_tensor)
result['out']   # 픽셀별 클래스 예측 점수
```


### 모델 종류에 따른 리턴 타입 비교

- 모델은 일반적으로 `dict`(딕셔너리) 를 반환
- `model()`은 PyTorch에서 모델을 호출하는 공통 방식이지만, 리턴 타입까지 공통으로 정해진 것은 아닙니다. 따라서 새 모델을 사용할 때는 해당 모델의 문서나 `forward()` 구현을 확인하는 습관이 중요합니다.

|모델 종류|예시 리턴 타입|반환 데이터|
|---|---|---|
|이미지 분류 CNN|`torch.Tensor`|클래스별 점수|
|ResNet18|`torch.Tensor`|클래스별 점수|
|객체 검출 모델|모델에 따라 다름|바운딩 박스, 클래스, 신뢰도 등|
|DeepLabV3|`dict`|Segmentation 점수 Tensor|
|사용자 정의 모델|직접 구현한 타입|구현한 `forward()`의 반환값|

### DeepLabV3에서 클래스 확인하기

```python
        # 클래스 확인
        classes = self.weights.meta["categories"]
        print("클래스 수:", len(classes))
        print(classes)
```


### 학습 데이터셋 Pascal VOC 과 COCO 관계

- - 일반적으로 VOC는 20개 객체 클래스를 사용하고, COCO의 대표적인 객체 검출 과제는 80개 클래스를 사용합니다. 이미지 수와 평가 방식도 다릅니다

###### 데이터셋 관계
```
COCO 데이터셋
      : 다양한 객체와 장면을 포함하는 대규모 데이터
      
     ↓
     
VOC와 겹치는 20개 객체 클래스 선택
      : 비행기, 자전가, 자동차, 사람 등
      
     ↓
     
COCO 데이터로 학습한 VOC 클래스용 모델
      : 배경 포함 21개 클래스로 Segmentation 예측     

```

> **이미지와 정답 라벨은 COCO에서 가져온 것이며, 선택한 클래스와 라벨은 VOC 기준에 맞춰서 사용**


#### Pascal VOC 클래스 21개

|클래스 ID|영문 클래스명|한글 의미|
|---|---|---|
|0|`background`|배경|
|1|`aeroplane`|비행기|
|2|`bicycle`|자전거|
|3|`bird`|새|
|4|`boat`|보트·배|
|5|`bottle`|병|
|6|`bus`|버스|
|7|`car`|자동차|
|8|`cat`|고양이|
|9|`chair`|의자|
|10|`cow`|소|
|11|`dining table`|식탁|
|12|`dog`|개|
|13|`horse`|말|
|14|`motorbike`|오토바이|
|15|`person`|사람|
|16|`potted plant`|화분|
|17|`sheep`|양|
|18|`sofa`|소파|
|19|`train`|기차|
|20|`tv/monitor`|TV·모니터|

##### PASCAL VOC란
- PASCAL Visual Object Classes의 약자
- 정확히는 PASCAL VOC(Visual Object Classes) Challenge 라는 컴퓨터 비전 연구 프로젝트 및 벤치마크를 가리킵니다.
- 이미지 속에 있는 자동차, 사람, 자전거 등의 객체를 컴퓨터가 인식하고 구분하는 성능을 평가하기 위한 **데이터셋과 평가 체계**

	- PASCAL: Pattern Analysis, Statistical Modelling and Computational Learning의 약자입니다.
	- Visual: 시각적인, 이미지와 관련된
	- Object: 객체, 물체
	- Classes: 클래스, 종류

> **VOC는 단순히 데이터셋의 이름만 의미하는 것이 아니라, 객체 검출과 시맨틱 세그멘테이션 등의 성능을 평가하기 위해 널리 사용된 벤치마크를 의미합니다.**


### 1.  cv1028_1_segmentation.ipynb

##### 전체소스

```python
import io
import requests
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 도로 이미지에서 차량, 보행자 객체를 픽셀 단위로 분할하는 클래스
# COCO 데이터로 학습한 VOC 클래스용 모델
# Pascal VOC / COCO 데이터셋으로 사전 학습된 DeepLabV3 모델 사용
class reodSceneSegmenter:

    #-------------------------------------------
    # step 1: 환경 설정 및 모델 로드 (Initialization)
    # 생성자: device 선택, Model, Weights 다운로드, 전처리기 설정, 클래스 팔레트 지정
    #-------------------------------------------
    def __init__(self, device=None):

        # 연산장치 설정(gpu/cpu)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 사전 학습된 가중치 및 모델 DEFAULT = COCO_WITH_VOC_LABELS_V1
        self.weights = DeepLabV3_ResNet101_Weights.DEFAULT   
        self.model = deeplabv3_resnet101(weights=self.weights).to(self.device)
        # 모델을 추론 모드(Evaluation Mode)로 설정하여 BatchNorm 오류를 방지합니다.
        self.model.eval()

        # 모델에 맞는 표준 전처리
        self.preprocess = self.weights.transforms()

        # 클래스 확인
        classes = self.weights.meta["categories"]
        print("클래스 수:", len(classes))
        print(classes)

        # 시각화를 위한 클래스별 색상 정의()
        # 현재 모델의 가중치에 대응하는 클래스 목록과 인덱스를 사용해야 함
        self.palette = np.zeros((21,3), dtype=np.uint8)
        self.palette[0] = [0,0,0] # 배경: 검정
        self.palette[1] = [0,128,255] # 배경:하늘색
        self.palette[2] = [0,255,0] # 자전거(bicycle) 녹색
        self.palette[6] = [255,255,0] # 버스(Bus) 노랑
        self.palette[7] = [255,0,0] # 자동차(Car): 빨강
        self.palette[14] = [255,128,0] # 오토바이(Motorbike): 주황
        self.palette[15] = [0,255,255] # 사람/보행자(Person) : Cyan(청록)

    #-------------------------------------------
    # Step 2: 이미지 입력 및 로드 (Image Loading)
    # 로컬 파일 또는 웹 URL에서 이미지 가져옴
    #-------------------------------------------
    def load_image(self, source:str) -> Image.Image:
        if source.startswith('http://') or source.startswith('https://'):
            headers = {"User-Agent":'Mozilla/5.0'}
            response = requests.get(source, headers=headers, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        else:
            #로컬 환경의 파일에서 이미지 불러옴
            image = Image.open(source).convert('RGB')
        return image

    #-------------------------------------------
    # 입력된 이미지 대해서 Segmentic Segmentation 수행
    #-------------------------------------------
    def predict(self, image:Image.Image) -> np.ndarray:
        # 배치 차원 추가해 지정된 연산 장치로 전달
        #(c, h, w) -> (b, c, h, w) : (1, c, h, w)
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 기울기 계산 사용자 않음 -> 메모리 절약 및 추론 성능 높임
        with torch.no_grad():
            # forward() 실행 -> 결과딕셔너리에서 'out' 참조(픽샐별 클래스 예측 점수)  -> [0] 첫번째 이미지선택
            output = self.model(input_tensor)['out'][0]

        # 각 픽셀마다 가장 높은 확률값을 가지는 클래스 인덱스 추출. 모델의 출력 Tensor를 픽셀별 클래스 번호로 변환
        mask = output.argmax(0).byte().cpu().numpy()

        return mask

    #-------------------------------------------
    # 원본 이미지에 모델이 분류한 객체 영역을 컬러 마스크를 사용하여 시각화 및 저장
    #-------------------------------------------
    def visualize(self, image:Image.Image, mask:np.ndarray, alpha:float=0.6, save_path:str=None):
        '''
        원본 이미지에 모델이 분류한 객체 영역을 컬러 마스크를 사용하여 시각화 및 저장

        parameters
            image: 원본 이미지
            mask: 클래스별 번호 매핑 ndarray
            alpha: 투명도
            save_path: 결과 이미지를 저장할 경로

        '''
        # 전처리 과정에서 달라진 이미지의 가로 세로 비율을 원래 크기로 복원 -> 모델의 마스크 크기가 (320, 240)이라면 원본 이미지에 맞게 (640, 480)으로 변경
        # fromarray(mask) : PIL 형식으로 변환
        # Image.NEARST : 확대할 때 클래스 정보가 변하지 않도록 인접보간법(NEARST) 사용
        #               - 정수값이 변하지 않도록 하기 위함!
        mask_img = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
        mask_resized = np.array(mask_img)   # PIL 이미지를 Numpy 배열로 변환

        # 클래스 정수 ID 와 색상값과 1:1 매핑시킴
        color_mask = self.palette[mask_resized]

        # 이미지를 numpy 배열로 변환하여 수치 연산과 합성 처리
        image_np = np.array(image)
        blended = image_np.copy()

        # 배경(0)을 제외한 영역 탐지
        object_area = mask_resized > 0

        # 반투명 합성 처리
        blended[object_area] = ( image_np[object_area] * (1-alpha) + color_mask[object_area] * alpha ).astype(np.uint8)

        # 1번째 서브플롯: 원본 이미지
        plt.figure(figsize=(18,6))
        plt.subplot(1,3,1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        # 2번째 서브플롯: 모델 추출 Mask
        plt.subplot(1,3,2)
        plt.title('Segmentation Mask(Red:Car, Cyan:Person)')
        plt.imshow(color_mask)
        plt.axis('off')

        # 3번째 서브플롯: 원본 + Mask = 반투명 합성
        plt.subplot(1,3,3)
        plt.title('Overlay Result')
        plt.imshow(blended)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"결과 이미지 저장: {save_path}")


if __name__ == '__main__':
    #image_path = "https://github.com/pytorch/hub/raw/master/images/deeplab1.png"
    #image_path = '/content/drive/MyDrive/Colab Notebooks/dataset/images/person1.jpg'
    image_path = 'https://cdn.pixbay.com/photo/'
    seg = reodSceneSegmenter()

    image = seg.load_image(image_path)
    mask = seg.predict(image)
    seg.visualize(image, mask)

```

##### 파이프라인

#### **Step 1: 환경 설정 및 모델 로드 (Initialization)**

- 사용 가능한 연산 장치(GPU가 있으면 CUDA, 없으면 CPU)를 자동으로 감지하여 설정합니다.
- `deeplabv3_resnet101` 모델과 사전 학습된 가중치(`DeepLabV3_ResNet101_Weights.DEFAULT`)를 불러온 후 추론 모드(`eval()`)로 전환합니다.
- 시각화 과정에서 각 객체(클래스)를 서로 다른 색상으로 구분하기 위해 클래스별 고유 색상 팔레트(Color Palette)를 정의합니다.

#### **Step 2: 이미지 입력 및 로드 (Image Loading)**

- 웹 URL 주소 또는 로컬 파일 경로로부터 분석할 원본 이미지를 불러옵니다 (`load_image` 메서드).
- 다양한 포맷의 이미지를 일관되게 처리하기 위해 RGB 채널 형식으로 변환합니다.

#### **Step 3: 데이터 전처리 (Preprocessing)**

- 모델에 입력하기 전, 모델 학습 당시와 동일한 기준(크기 조정, 텐서 변환, 정규화 등)으로 이미지를 변환합니다 (`self.preprocess`).
- PyTorch 모델이 요구하는 4차원 배치 형태 `(Batch, Channel, Height, Width)`로 차원을 확장합니다 (`unsqueeze(0)`).

#### **Step 4: 모델 추론 및 예측 (Prediction)**

- 기울기 계산을 비활성화(`torch.no_grad()`)하여 메모리를 절약하고 속도를 높인 상태로 모델에 이미지를 입력합니다.
- 모델의 출력값(각 픽셀별 클래스 예측 점수) 중 가장 높은 확률을 가진 클래스 번호(ID)를 추출하여 2차원의 **세그멘테이션 마스크(Segmentation Mask)**를 생성합니다 (`argmax(0)`).

#### **Step 5: 시각화 및 결과 저장 (Visualization & Save)**

- 추론 결과로 나온 마스크를 원본 이미지 크기에 맞게 인접 보간법(Nearest Neighbor)을 사용하여 복원합니다.
- 미리 정의한 색상 팔레트를 마스크에 매핑하여 컬러 마스크를 만듭니다.
- 원본 이미지와 컬러 마스크를 반투명(`alpha`)하게 합성하여 객체 위치를 직관적으로 보여주는 오버레이 이미지를 생성하고, 원본·마스크·합성 결과 3가지 서브플롯으로 시각화한 후 필요 시 파일로 저장합니다.



### 2. cv_1028_2_Segmentagion Evaluate.ipynb

##### 전체소스
```python

# Confusion Maxrix(혼동행렬) 기반으로 평가(PA, IoU, Dice)하는 클래스
# 혼동(Confusion): 정답과 예측이 일치하는 경우와 서로 어긋나는 경우를 교차해서 정리한다
#    교차          실제 클래스와 예측 클래스 사이의 혼동을 기록한다
#                  모델이 실제 클래스와 예측 클래스 사이에서 어떤 판단을 했는지 교차표 형태로 보여주기
class segmentationEvalutor():

    def __init__(self, num_classes:int = 21, class_names:dict=None):
        self.num_classes = num_classes   # 클래스 총 개수
        self.class_names = class_names   # 클래스 정보

    #--------------------------------------------------------------
    # 예측 마스크와 정답 마스크를 비교하여 혼동 행렬을 고속으로 계산
    #--------------------------------------------------------------    
    def __compute_confusion_matrix(self, pred_mask:np.ndarray, gt_mask:np.ndarray) -> np.ndarray:
        '''
        예측 마스크와 실제 정답 마스크를 비교하여 고속으로 2차원 혼동 행렬(Confusion Matrix)을 
        계산하고 반환합니다.

        parameters
            pred_mask (np.ndarray): 모델이 예측한 클래스 인덱스 맵 (2D 배열)
            gt_mask (np.ndarray): 실제 정답(Ground Truth) 클래스 인덱스 맵 (2D 배열)

        returns
            hist (np.ndarray): (num_classes, num_classes) 크기를 가지는 2차원 혼동 행렬
        '''
        # 정답 레이블 범위 내의 유효한 값(0~num_classes-1)에 포함하는 픽셀만 처리
        valid = (gt_mask >= 0) & (gt_mask < self.num_classes)

        # 정답클래스 + 전체 클래스 수 + 예측클래스 -> 1차원 결합 인덱스 배열을 구함
        encoded_indices = self.num_classes * gt_mask[valid].astype(int) + pred_mask[valid].astype(int  )

        # bincount: 정답*클래스 수 + 예측연산 -> 2차원 좌표를 1차원 고유 인덱스값으로 변환
        # 반복문 사용없이 bincount만 사용하여 모든 클래스 쌍의 빈도수를 매우 빠르게 연산
        # 클래스가 3개라면 실제 클래스와 예측 클래스의 조합 9가지를 0~8이라는 서로 다른 숫자로 표현
        hist = np.bincount(self.num_classes * gt_mask[valid].astype(int) + pred_mask[valid], 
                           minlength=self.num_classes ** 2
                ).reshape(self.num_classes, self.num_classes)

        return hist

    #------------------------------------------------------------
    # PA(Pixel Accuracy), IoU, Dice 를 계산하여 결과를 딕셔너리로 변환
    #------------------------------------------------------------
    def evaluate(self, pred_mask:np.ndarray, gt_mask:np.ndarray):
        '''
        예측 마스크와 실제 정답 마스크의 해상도를 일치시킨 후 전체 픽셀 정확도(PA), 평균 IoU(mIoU), 
        평균 Dice 계수(mDice), 그리고 클래스별 세부 지표를 계산하여 딕셔너리로 반환합니다.

        parameters
            pred_mask (np.ndarray): 모델이 예측한 클래스 인덱스 맵
            gt_mask (np.ndarray): 실제 정답 클래스 인덱스 맵

        returns
            metrics (dict): 다음과 같은 구조를 가진 평가지표 결과 딕셔너리

        '''
        # pred_mask 와 gt_mask의 shape가 일치하지 않으면 gt_mask의 해상도 기준으로 예측값 크기를 맞춤
        if pred_mask.shape != gt_mask.shape:
            pred_img = Image.fromarray(pred_mask).resize(gt_mask.shape, resample=Image.NEAREST)
            pred_mask = np.asarray(pred_img)

        # 혼동 행렬 처리
        hist = self.__compute_confusion_matrix(pred_mask, gt_mask)

        # PA(Pixel Accuracy)
        pixel_acc = np.diag(hist).sum() / (hist.sum() + 1e-10)

        # IoU(Intersection over Union)
        # intersection: 혼동 행력의 대각 성분
        # union: 실제 정답 분포 + 예측값 분표 - 교집합
        intersection = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)

        # Dice score: 정답 마스크와 예측 마스크가 겹치는 정도를 평가
        # Dice = 2 * 교집합 / (실제정답픽셀수 + 예측한 픽셀수)        
        dice = 2 * intersection / (hist.sum(axis=1) + hist.sum(axis=0) + 1e-10)

        # 실제로 존재하는 클래스 정보
        present_classes = union > 0

        # 실제 존재하는 클래스들에 대해서 IoU와 dice의 평균값 구함
        miou = np.mean(iou[present_classes])
        mdice = np.mean(dice[present_classes])

        # 개별 클래스 단위 세부 지표
        class_metrics = {}
        for cls_idx in np.where(present_classes)[0]:
            name = self.class_names.get(cls_idx, f"Class_{cls_idx}")
            class_metrics[name] = {
                "IoU": float(iou[cls_idx]),
                "Dice": float(dice[cls_idx]),
                'GT_Pixel': int(hist.sum(axis=1)[cls_idx]),
                'Pred_Pixel': int(hist.sum(axis=0)[cls_idx]),
            }

        return {
            'PA': float(pixel_acc),
            'mIoU': float(miou),
            'mDice': float(mdice),
            'Class_Metrics': class_metrics
        }
    
    #------------------------------------------------------
    # 결과 데이터 출력
    #------------------------------------------------------
    def print_report(self, metrics):
        '''
        evaluate 메서드가 반환한 평가지표 결과를 포맷에 맞춰 콘솔 화면에 가독성 있게 출력합니다.

        parameters
            metrics (dict): evaluate 함수에서 반환한 평가지표 딕셔너리

        '''
        print(f'PA: {metrics["PA"]:.4f}')
        print(f'mIoU: {metrics["mIoU"]:.4f}')
        print(f'mDice: {metrics["mDice"]:.4f}')
        for cls_name, data in metrics['Class_Metrics'].items():
            print(f'  {cls_name}:')
            print(f'\tIoU: {data["IoU"]:.4f}')
            print(f'\tDice: {data["Dice"]:.4f}')
            print(f'\tGT_Pixel: {data["GT_Pixel"]}')
            print(f'\tPred_Pixel: {data["Pred_Pixel"]}')


    #-------------------------------------------
    # 정답 마스크 라벨링 파일이 없을 경우 가상의 정답 데이터 생성
    #-------------------------------------------
    def create_mock_ground_truth(self, pred_mask:np.ndarray, shift_pixels=4, noise_ratio=0.02):
        '''
        실제 정답 데이터가 없을 경우를 대비하여, 예측 마스크에 미세한 평행 이동(shift)과 
        무작위 노이즈를 주어 가상의 정답(Ground Truth) 마스크를 생성합니다.

        parameters
            pred_mask (np.ndarray): 기준이 될 예측 마스크
            shift_pixels (int): 가로/세로 방향으로 이동시킬 픽셀 크기 (기본값: 4)
            noise_ratio (float): 노이즈가 발생해 배경(0)으로 덮어씌워질 비율 (기본값: 0.02)

        returns
            mock_gt (np.ndarray): 변형 처리된 가상의 정답 마스크 배열

        '''
        # 가로, 세로 축을 기준으로 평행 이동 오차 발생
        mock_gt = np.roll(pred_mask, shift=shift_pixels, axis=0)
        mock_gt = np.roll(mock_gt, shift=shift_pixels, axis=1)

        # 지정된 오차 발생 비율에 맞춰 노이즈 마스크를 생성하여 배경 클래스(0)로 덮어 씌움
        noise_mask = np.random.rand(*pred_mask.shape) < noise_ratio
        mock_gt[noise_mask] = 0

        return mock_gt
```

##### 파이프라인

이 프로그램은 크게 **3단계(입력 및 추론 ➡️ 평가 지표 계산 ➡️ 시각화)**의 파이프라인으로 구성되어 작동합니다.

#### **Step 1: 데이터 로드 및 모델 추론 (Inference)**

- **클래스**: `reodSceneSegmenter`

1. **이미지 로딩 (`load_image`)**: 웹 URL 또는 로컬 경로에서 이미지를 PIL Image 형식으로 읽어옵니다.
2. **전처리 및 추론 (`predict`)**: DeepLabV3 모델 가중치(`DeepLabV3_ResNet101_Weights.DEFAULT`)에 맞는 정규화 및 크기 변환 후, 모델에 입력하여 각 픽셀별로 확률이 가장 높은 클래스 인덱스를 추출해 `pred_mask`를 생성합니다.

#### **Step 2: 평가지표 계산 (Evaluation)**

- **클래스**: `segmentationEvalutor`

1. **가상 정답 생성 (`create_mock_ground_truth`)**: 실제 정답 데이터가 없을 경우를 대비해, 예측한 마스크(`pred_mask`)를 미세하게 평행 이동(shift)시키고 무작위 노이즈를 섞어 모조 정답 데이터(`gt_mask`)를 생성합니다.
2. **혼동 행렬 계산 (`__compute_confusion_matrix`)**: 반복문 없이 NumPy의 `bincount` 연산만을 활용하여 고속으로 실제 클래스와 예측 클래스 간의 빈도를 기록한 2차원 Confusion Matrix를 계산합니다.
3. **지표 연산 (`evaluate` & `print_report`)**:
    - **PA (Pixel Accuracy)**: 전체 픽셀 중 맞춘 픽셀의 비율
    - **mIoU (Mean Intersection over Union)**: 클래스별 교집합/합집합 비율의 평균
    - **mDice (Mean Dice Coefficient)**: 두 마스크의 겹침 정도를 나타내는 Dice 계수의 평균
    - 이 지표들을 전체 및 개별 클래스 단위(`__background__`, `bus`, `car`, `person` 등)로 계산하여 콘솔에 보기 좋게 출력합니다.

#### **Step 3: 결과 시각화 (Visualization)**

- **클래스**: `reodSceneSegmenter` (메서드: `visualoze_with_eval`)
- **4분할 플롯 시각화**:
    1. **Original Image**: 원본 입력 이미지
    2. **Ground Truth (Mock) Mask**: 가상으로 만든 실제 정답 영역 (클래스별 색상 매핑)
    3. **Predicted Mask**: 모델이 예측한 클래스 영역과 예측 성능 지표(mIoU) 표시
    4. **Error Map**: 예측값과 정답값이 불일치하는 영역을 자홍색(Magenta, `[255, 0, 255]`)으로 뚜렷하게 표시하여 직관적으로 모델의 오차를 분석할 수 있게 돕습니다.



### 소스 분석

####  두 클래스 ID를 하나로 결합하여 픽셀 단위 혼동행렬 만들기

__compute_confusion_matrix()함수

```python
hist = np.bincount(
           self.num_classes * gt_mask[valid].astype(int) + pred_mask[valid], 
           minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
```

- 수식  $\text{index} = C \times GT + Pred$    
	- $C$ 는 클래스 수
	- GT : gt_mask(정답 마스크)
	- Pred : pred_mask(예측 마스크)
	- gt_mask 에 클래스수에 해당하는 진수 가중치를 지정

- 클래스가 3개라면  실제 클래스와 예측 클래스의 조합 9가지를 `0~8`이라는 서로 다른 숫자로 표현합니다.

| 실제 GT | 예측 Pred | 계산                 | 결합 인덱스 |
| ----- | ------- | ------------------ | ------ |
| 0     | 0       | $(3 \times 0 + 0)$ | 0      |
| 0     | 1       | $(3 \times 0 + 1)$ | 1      |
| 0     | 2       | $(3 \times 0 + 2)$ | 2      |
| 1     | 0       | $(3 \times 1 + 0)$ | 3      |
| 1     | 1       | $(3 \times 1 + 1)$ | 4      |
| 1     | 2       | $(3 \times 1 + 2)$ | 5      |
| 2     | 0       | $(3 \times 2 + 0)$ | 6      |
| 2     | 1       | $(3 \times 2 + 1)$ | 7      |
| 2     | 2       | $(3 \times 2 + 2)$ | 8      |
- `np.bincount()` - 각 조합의 등장 횟수 계산
- `minlength=9`는 등장한 인덱스가 적더라도 결과 배열의 길이를 최소 9로 유지합니다.
- .reshape(self.num_classes, self.num_classes)` — 2차원 혼동행렬 생성


예시: 행은 실제 정답(GT), 열은 모델 예측(Pred)

| 실제 \ 예측 | 배경(0) | 자동차(1) | 사람(2) |
| ------- | ----- | ------ | ----- |
| 배경(0)   | 90    | 5      | 5     |
| 자동차(1)  | 3     | 15     | 2     |
| 사람(2)   | 4     | 1      | 10    |

- 대각선 값: 정답과 예측이 일치한 픽셀 수
- 대각선 밖의 값: 다른 클래스로 잘못 예측한 픽셀 수

#### PA(Pixel Accuracy), IoU, Dice 를 계산하여 결과를 딕셔너리로 변환


evaluate(self, pred_mask:np.ndarray, gt_mask:np.ndarray) 함수

###### PA 계산

주의할 점은 배경 픽셀이 압도적으로 많으면 배경을 잘 맞히는 것만으로도 PA가 높아질 수 있다는 것입니다.

- np.diag(hist)`)`: 혼동 행렬의 대각선 값 추출
- `hist.sum()`: 전체 픽셀 수


##### IoU(Intersection over Union)
- hist.sum(axis=1)  : 클래스별 실제 정답 픽셀 수 (열단위합)
- `hist.sum(axis=0)`:  클래스별 예측 픽셀 수 (행단위합)