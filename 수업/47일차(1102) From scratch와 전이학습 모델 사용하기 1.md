


## 수업목표
- 프로젝트: 재활용품을 분리하는 딥러닝 모델을 훈련한 후 활용할 수 있게 코렙과 윈도우에서 개발
- 직접 만든 심층신경망(From Scratch)과 전이학습(Transfer Learning)으로 만든 모델 만들고 사용하기
- 

## 실습파일
- Model_Class_From _the_Scratch.py                   <-  From Scratch(직접만든 신경망)
- Model_Class_Transfer_Learning_MobilNet.py      <- Transfer Learning(전이학습)
- Dataset_Class.py                                          <- 파이토치 커스텀 데이터세트 클래스

### 클래스 구성

| 클래스명                                   | 용도                         | 파일명                                        | 시간  |
| -------------------------------------- | -------------------------- | ------------------------------------------ | --- |
| PyTorch_Custom_Dataset_Class           | 데이터세트                      | Dataset_Class.py                           |     |
| MobileNet                              | 전이학습을 활용한 심층 신경망           | Model_Class_Transfer _Learning_MobilNet.py |     |
| Model_From_Scratch                     | 직접 신경망을 구현                 | Model_Class_From _the_Scratch.py           |     |
| Pytorch_Classification_Tranining_Class | 심층 신경망 트레이닝(위의 3개 클래스를 이용) | Training_Class.py                          |     |
|                                        |                            |                                            |     |

**전이학습**의 가장 큰 장점은 반드시 학습 속도가 빠르다는 것이 아니라, 이미 학습된 특징을 활용하여 **적은 데이터와 학습으로도 좋은 성능**을 얻을 수 있다는 것입니다.

##### Dataset_Class.py

```python
# filename : Dataset_Class.py
# 파이토치 커스텀 데이터세트 클래스
# 데이터 세트를 읽어오기 위한 데이터 세트 클래스 상속 : PyTorch Custom Dataset Class
#
# 생성자 : __init__()
# 오버라이딩:  __len__(), __getitem__()
# 일반메서드: __save_label_map__()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image 
import shutil  # 폴더 삭제
import stat    # 파일 상태 변경
import numpy as np
import matplotlib.pyplot as plt


class PyTorch_Custom_Dataset_Class(Dataset):

    # 파이토치의 데이터세트 클래스를 상속받은 후 초기화
    def __init__(self,
                 dataset_dir='/content/Recycle_Classfication_Dataset',
                 transform=None          
        ):
        super().__init__()

        # 재활용 분류 데이터셋 받기. 데이터셋 폴더가 없으면 github에서 재활용 분류 데이터셋 받기
        if not os.path.isdir(dataset_dir):           
            dataset_dir = os.path.join(os.getcwd(), 'Recycle_Classification_Dataset')
        if not os.path.isdir(dataset_dir):
            os.system('git clone https://github.com/JinFree/Recycle_Classification_Dataset.git')

        # git 폴더 제거 ->       # 리눅스에서만 동작는 명령을 윈도우에서도 동작하도록 수정함
        #os.system('rm -rf ./Recycle_Classfication_Dataset/.git')  
        git_dir = os.path.join(dataset_dir, '.git')
        if os.path.isdir(git_dir):
            shutil.rmtree(git_dir, onerror=lambda func, path, _:                 # func <- 삭제에 실패했던 작업 함수와 파일 경로
                                                (os.chmod(path, stat.S_IWRITE),  # 파일에 쓰기 권한 부여
                                                func(path)))                     # 실패했던 작업 다시 실행  

        # 데이터셋 루트 경로
        self.image_abs_path = dataset_dir

        # 사용자 정의 전처리
        self.transform = transform
        # 별도로 입력된 전처리 방법이 없는 경우 사전에 입력한 전처리 방법을 선언
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),         # 256 크기로 리사이징
                    transforms.RandomCrop(224),     # 224 크기로 랜덤하게 증강
                    transforms.ToTensor(),          # 텐서로 변환
                    transforms.Normalize(           # 이미지넷 평균/표준편차 정규화 
                        mean=[0.485, 0.456, 0.406],   # ImageNet 데이터셋의 RGB 채널별 통계값으로 이미 정해져 있음
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        # 하위 폴더 이름을 클래스 목록으로 사용하고 숨김 폴더(.으로 시작)는 제외
        self.label_list = [ 
            d for d in os.listdir(self.image_abs_path) 
            if os.path.isdir(os.path.join(self.image_abs_path, d)) and not d.startswith(".")
        ]

        # 클래스 이름의 리스트를 정렬
        self.label_list.sort()
        self.x_list = []      # 이미지 파일 경로 목록      
        self.y_list = []      # 클래스 인덱스 목록

        # 폴더내의 모든 파일의 경로를 리스트로 저장하고 클래스 인덱스를 y 리스트에 저장
        for label_index, label_str in enumerate(self.label_list):
            img_path = os.path.join(self.image_abs_path, label_str)
            img_list = os.listdir(img_path)

            for img in img_list:
                full_img_path = os.path.join(img_path, img)
                if os.path.isfile(full_img_path) :
                    self.x_list.append(full_img_path)
                    self.y_list.append(label_index)


    # 데이터세트의 크기 혹은 길이를 반환
    def __len__(self):
        return len(self.x_list)

    # 이미지와 정답을 불러옴
    def __getitem__(self, idx):
        # PIL 모듈의 Image 함수를 활용해 이미지를 읽기
        image = Image.open(self.x_list[idx])

        # 이미지가 RGB가 아닌 경우 RGB 이미지로 바꿈. 흑백 이미지는 예외 처리
        if image.mode  != 'RGB':
            image = image.convert("RGB")

        # 데이터 세트를 신경망에 입력하기 전에 전처리를 수행
        if self.transform is not None:
            image = self.transform(image)
        
        # 전처리된 이미지, 라벨 인덱스
        return image, self.y_list[idx]

    # 라벨 인덱스 순서대로 클래스 이름을 txt 파일에 한 줄씩 저장
    def __save_label_map__(self, dst_text_path='lable_map.txt'):
        label_list = self.label_list
        f = open(dst_text_path, 'w')
        for i in range(len(label_list)):
            f.write(label_list[i] + "\n")
        f.close()

    # 분류할 클래스 개수
    def __num_classes__(self):
        return len(self.label_list)

if __name__ == '__main__':
    dataset = PyTorch_Custom_Dataset_Class()
    dataset.__save_label_map__()
```



##### Model_Class_Transfer_Learning_MobilNet.py

전이학습을 활용한 심층 신경망 구현

```python
# filename : Model_Class_Transfer_Learning_MobilNet.py
# 전이학습을 활용한 심층 신경망 구현
# 교재 5장 참조
# 전이학습 : MobilNetV2 사용
# 

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # ImageNet 학습 가중치 적용한 모델을 불러옴
        self.network = models.mobilenet_v2(pretrained=pretrained)

        # MobileNet의 마지막 레이어가 num_classes 만큼의 클래스를 분류할 수 있게 수정
        # 기존 출력층의 입력 특징 가져오기: MobileNet2 1280 출력 채널
        num_ftrs = self.network.classifier[1].in_features
        # 출력층 교체 num ftrs -> num classes
        self.network.classifier[1] = nn.Linear(num_ftrs, num_classes)
        self.classifier = nn.Sequential(nn.Softmax(dim=-1))

    # 순전파
    def forward(self, x):
        x = self.network(x)      # MobileNetV2 합성곱층
        x = self.classifier(x)  # 사용자 정의 분류기

        return x
```

-  이미지넷의 1000개 클래스의 이미지 1백만개를 학습 데이터로 사용해 사전 학습된 신경망과 가중치를 받아서 학습을 수행
- 파이토치를 비롯한 많은 딥러닝 프레임워크에서 이미지 분류를 위한 신경망과 이미지넷 데이터를 활용해 학습한 가중치를 제공

##### Model_Class_From_the_Scratch.py

```python
# filename : Model_Class_From_the_Scratch.py
# 교재 5장 참조
# 1. 직접 신경망을 구현하여 추론
# 2. 전이학습(이 방식을 주로 사용할 것임)

import cv2 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import ImageStat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# nn.Module을 상속받아 사용자 CNN 모델 클래스 정의
class Model_From_Scratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            # 첫번째 합성곱층  입력 3채널(노드) -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),  # 정규화
            nn.ReLU(),           # 활성화 

            # 두번째 합성곱층  채널 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 세번째 합성곱층  채널 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 각 채널을 하나의 값으로 압축
            nn.AdaptiveAvgPool2d(1),   # 전역평균 풀링. maxPooling보다 좋음
            
            # 첫번째 완전연결층 
            nn.Flatten(),              # 1차원 벡터 처리
            nn.Linear(128, 512),       # 완전 연결층
            nn.ReLU(),

            # 2번째 완전연결층
            nn.Dropout(),              # 과적합 방지
            nn.Linear(512, 64),
            nn.ReLU(),

            # 3번째 완전연결층
            nn.Dropout(),
            nn.Linear(64, num_classes),

            # 클래스별 확률 변환(0~1)
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        return self.classifier(x)

```

- \__init()__ 와 forward() 로 구성
- \__init()__ 에서는 forward()에서 사용할 신경망 정의
- forward()에서는 정의한 신경망 구조에 따라 계산을 수행한 후 분류 결과를 반환


###### 저장된 모델 파일 비교

> 저장된 모델 Pytorch_Train_From_Scratch.pt 는 **800KB** 정도이고  Pytorch_Transfer_Learning_Mobilenet.pt 는 **8MB** 로 나옴 

