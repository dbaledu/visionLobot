# 1. OpenCV 전체 교육 커리큘럼

OpenCV 수업은 단순히 이미지 처리 라이브러리의 함수를 익히는 것보다 이미지를 읽고 분석하는 방법부터 시작해서 전통적인 컴퓨터 비전, 딥러닝 기반 영상 인식, 최신 Vision Transformer, 객체 검출까지 단계적으로 발전시키는 방식으로 구성하는 것이 좋습니다.

특히 자율주행, AMR/AGV, 영상 분석 등의 프로젝트까지 연결할 수 있도록 커리큘럼을 설계하겠습니다.

## Computer Vision & Deep Learning 과정

기초 이미지 처리 → 전통적 영상 처리 → 딥러닝 → 최신 객체 인식

1. 1단계. OpenCV 기초
    이미지 구조, 픽셀, 채널, 이미지 입출력
    
2. 2단계. 이미지 변환 및 전처리
    Resize, Crop, Rotate, Color Space, Threshold
    
3. 3단계. 이미지 필터링 및 특징 추출
    Blur, Edge, Morphology, Contour, Hough
    
4. 4단계. 이미지 분석 및 영상 처리
    Histogram, ROI, Object Tracking, Video Processing
    
5. 5단계. CNN 기초
    Convolution, Pooling, Feature Map, Classification
    
6. 6단계. 전이학습
    Pretrained Model, Fine-tuning, MobileNet, ResNet
    
7. 7단계. 이미지 세그멘테이션
    Semantic, Instance, Panoptic Segmentation
    
8. 8단계. Vision Transformer
    Attention, Patch Embedding, ViT, DETR
    
9. 9단계. YOLO 객체 검출
    Object Detection, Bounding Box, NMS, YOLO 학습
    
10. 10단계. 종합 프로젝트
    자율주행 영상 분석 및 객체 인식 시스템
    

## 교육 기간 제안

|구분|교육 기간|주요 내용|
|---|---|---|
|OpenCV 기초|1~2주|이미지 처리 및 변환|
|전통적 CV|1주|필터링, 특징 추출|
|CNN|1주|딥러닝 이미지 분류|
|전이학습|1주|사전학습 모델 활용|
|세그멘테이션|1주|픽셀 단위 객체 분석|
|Transformer|1주|Vision Transformer|
|YOLO|1~2주|객체 검출 및 학습|
|프로젝트|1주|통합 영상 분석|

총 9~10주 과정을 추천합니다. Python과 NumPy 기초를 이미 학습했다는 전제입니다.

# 2. 단계별 상세 커리큘럼

각 단계는 요청한 형식인 수업목표 → 수업내용 → 요약 → 실습과제로 정리했습니다.

## 1단계. OpenCV 기초 및 이미지 구조

기초 과정 : 권장 2일

[Digital Image Data Representation - Know the Code](https://images.openai.com/static-rsc-4/5XRfmlSyMeTCXk96nWz9HPOFJ5yceraUC_HFWOhUzvzVehloOLpYd54MFdiFVOQO4N4ZA9t0bKp8rdS1wSExF1vqaG89Mml9ftQ1hEi9WB-JQCEBLYCFVIGFNAe8OnEIhx1c7P0fXuZ_7F9fqRFcHoXu0pYcWHHVXV7vs1eP0M0?purpose=inline)

[cv2.imread()和cv2.cvtColor() 的使用-CSDN博客](https://images.openai.com/static-rsc-4/yUrPOvL6SrhxajYt3CiS9OkntFlozzqQkt-VCfImkWxTjPkniKI-EXrhGgeM9WKydZjY0MUEiDq9YOmmogDDOcHPx0Y2hlHUhz1fjWlITs6eNUNhNwfeXYghCWAaO9ci0l2zRpKhqARUoAPKOwgm8KMZX7P042A8RhjJabpiRTw?purpose=inline)

[[OpenCV/Python] 이미지 불러오기 & 읽기 & 저장하기](https://images.openai.com/static-rsc-4/rTMBwJ__o1yi_HLEzDuvb0SSnyQX1lKHBmFi_NyJxTEDHZTKHDn_eWkQptreLwESXmqfY-_PMouVoV6GDrO9cIhwhv7QsfW2C-Tw40a6-1_vliF0Hy7G-k2bnaxDQ3p5w0VsJL49GYzGZ3j70ezDTSSJPk8Wo-4_r9N4YKySsP0?purpose=inline)


① 수업목표

- 컴퓨터에서 이미지가 어떤 데이터 구조로 표현되는지 이해한다.
- OpenCV의 기본 사용법과 이미지 입출력 방법을 익힌다.
- NumPy 배열과 이미지 데이터의 관계를 이해한다.
    

② 수업내용

| 세부 주제     | 학습 내용                            |
| --------- | -------------------------------- |
| OpenCV 소개 | Computer Vision, OpenCV 특징       |
| 환경 구성     | opencv-python, NumPy, Matplotlib |
| 이미지 구조    | 픽셀, 해상도, 채널                      |
| 컬러 이미지    | RGB와 BGR 차이                      |
| 그레이스케일    | 1채널 이미지                          |
| 이미지 입출력   | imread(), imwrite(), imshow()    |
| 이미지 정보    | shape, dtype, size               |

주요 함수:

```python
import cv2import numpy as np
img = cv2.imread("image.jpg")
print(img.shape)
print(img.dtype)
print(img.size)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

③ 핵심 요약

- OpenCV는 기본적으로 BGR 순서로 컬러 이미지를 처리한다.
- 이미지는 NumPy의 ndarray 형태로 저장된다.
- 컬러 이미지는 일반적으로 `(높이, 너비, 3)` 구조이다.
- 픽셀은 0~255 범위의 밝기 또는 색상 값을 가진다.
    

④ 실습과제

1. 이미지 파일을 불러와 화면에 출력한다.
2. 이미지의 가로, 세로, 채널 수를 출력한다.
3. RGB와 BGR 이미지를 비교한다.
4. 이미지를 그레이스케일로 변환하여 저장한다.
5. NumPy 인덱싱으로 특정 픽셀의 색상값을 출력한다.
    
## 2단계. 이미지 변환 및 전처리

기초 과정 : 권장 2~3일

[Edit you image, resize, crop, flip, rotate, change format by Igornevzgliad | Fiverr](https://images.openai.com/static-rsc-4/Z-5VDtNyXiNdXbo5o3umKY_5zNjNt7VfWjMefmMq0jqDXbyWqAhp3HhfFjXLV_AT2nsTbAk2cUXigN10u6SAZy-VZPc52xOgBf5NcfdqTBVftbQK53y6_wbQhE70buwJ6b_2njnRjs4dod3CIMxT7zQ9il9LRu77tBp0-41sW_Q?purpose=inline)

[Invisible Cloak Using OpenCv. In childhood almost every one amazed… | by Syed Shoyab | Medium](https://images.openai.com/static-rsc-4/1zQ7HlK4sTQ9mxbOw_R_vdNT84T8L31TAs-sXdoZRN5Ql1-AX5o2n77KQJkLqh0oD66na5Ngzz0vGIsA3TDFs-y5CSM3FDf6fC3I0a4igMyjXKHdrZtXfsTRlmzXPThR4VJpYpX2TaFMlbtmIts8ze44h-zlMtbDccvzU4fJFys?purpose=inline)

[What is Thresholding in Image Processing? A Guide.](https://images.openai.com/static-rsc-4/pm0FsjVbwBuSI1urP9sASPZLZIAQlmJTRdgTKkV3aZOqBIs7S-ZXlFDLxKpa-FzKHvA5-r4XRQ6h1Or0SBL98XfoP_cOtcgduc1oMbvSJd8rc0bk7IfmhnUaDLPqeQZ5enVpVCUGwc53meTXJjHzBBbPFOGiFluLeuQcviGVJCY?purpose=inline)


① 수업목표

- 이미지의 크기와 방향을 변경할 수 있다.
- 다양한 컬러 공간의 특징을 이해한다.
- 이미지 분석에 필요한 기본 전처리 방법을 익힌다.

② 수업내용

| 세부 주제   | 학습 내용                    | 주요 함수                     |
| ------- | ------------------------ | ------------------------- |
| 크기 변환   | Resize, Interpolation    | resize()                  |
| 이미지 자르기 | Crop, ROI                | NumPy Slicing             |
| 회전 및 이동 | Rotate, Translation      | rotate(), warpAffine()    |
| 기하학적 변환 | Affine Transformation    | getAffineTransform()      |
| 원근 변환   | Perspective Transform    | getPerspectiveTransform() |
| 컬러 변환   | BGR, RGB, Gray, HSV, HLS | cvtColor()                |
| 밝기 조절   | Brightness, Contrast     | convertScaleAbs()         |
| 이진화     | Binary Threshold         | threshold()               |
| 적응형 이진화 | Adaptive Threshold       | adaptiveThreshold()       |

특히 다음 개념은 반드시 이해하도록 합니다.

- RGB/BGR: 색상 표현
- HSV: 색상(Hue), 채도(Saturation), 명도(Value)
- HLS: 색상(Hue), 밝기(Lightness), 채도(Saturation)
- Threshold: 특정 기준값을 이용한 픽셀 분류
    

③ 핵심 요약

이미지 변환은 크게 두 가지로 구분할 수 있습니다.

- 기하학적 변환: 이미지의 위치, 크기, 방향 변경
- 화소 변환: 픽셀의 밝기, 색상, 명암 변경
    

컬러 공간을 적절히 선택하면 차선, 신호등, 특정 색상 객체 등을 효과적으로 검출할 수 있다.

④ 실습과제

1. 하나의 이미지로 Resize, Crop, Rotate를 구현한다.
2. Affine 및 Perspective Transformation을 구현한다.
3. BGR 이미지를 HSV와 HLS로 변환한다.
4. HSV 범위를 이용해 노란색 객체만 추출한다.
5. Grayscale 이미지에 다양한 Threshold를 적용한다.
    

## 3단계. 이미지 필터링 및 특징 추출

전통적 Computer Vision : 권장 3일

[Image Manipulations in Python OpenCV](https://images.openai.com/static-rsc-4/WarPbwkn130U38TCuUy9zG00OuA58t5Icq67ncZEqQAv9RIZrig_A_Zxu3lHRVZQBsRZUmYCTHekuwf4fQ4yZprLxk60ruOrf-WQidyq5dMmIUqpNz8HAnz6WQZ3nzdoOAaTokXSnG82FkIvPgvxX_rlox73O6e9J6xnvpYHM2c?purpose=inline)

[Edge Detection of Image using OpenCV (CV2) in Python](https://images.openai.com/static-rsc-4/ent8vteTp9SxS331QlWy-6eaWm_NSTDGwX37_WkDzaLfMGAO6CCLSbdU-HQBbYEwuXtCg8mCQSEpbXrTXNqEZqWJItvtO5G7lOXqbh-LKoL5d-huQ6jTjMZAwW5y2tSl3mg4VgzqR26AI48uyj4R-0hKY6wM3FU61_2PXGNw_jA?purpose=inline)

[#006 Morphological transformations with OpenCV in Python](https://images.openai.com/static-rsc-4/Up73WZ9BdwKfcQUh6MddKIk7AUxPSstv3-p4CQamprkyvBt5JMi48HCraayesmg_OacZZ7nk7qUFl8PBr_b460VK3CXfZVjtGS4GXpaZCSMq-MZvsq_AZUnpO4VRWL_E7ERwgFG0WSMA87TH0fVqXagwlqW8IVb-KXnE12gooMg?purpose=inline)


① 수업목표

- 이미지의 노이즈를 제거하는 방법을 이해한다.
- 이미지의 경계선과 특징을 추출할 수 있다.
- 형태학적 연산을 이용해 객체의 형태를 분석한다.

② 수업내용

|세부 주제|학습 내용|주요 함수|
|---|---|---|
|평균 필터|Mean Blur|blur()|
|가우시안 필터|Gaussian Blur|GaussianBlur()|
|중간값 필터|Median Blur|medianBlur()|
|양방향 필터|Bilateral Filter|bilateralFilter()|
|에지 검출|Sobel, Laplacian|Sobel(), Laplacian()|
|에지 검출|Canny|Canny()|
|침식|Erosion|erode()|
|팽창|Dilation|dilate()|
|열림/닫힘|Opening, Closing|morphologyEx()|
|윤곽선|Contour Detection|findContours()|
|직선 검출|Hough Transform|HoughLinesP()|
|원 검출|Hough Circle|HoughCircles()|

③ 핵심 요약

이미지 분석의 기본 파이프라인을 이해하는 것이 중요합니다.
![[Pasted image 20261110151733.png]]

차선 및 직선 검출 결과

- 필터링은 노이즈를 줄이는 과정이다.
- Edge Detection은 이미지의 경계선을 추출한다.
- Morphology는 객체의 형태를 개선한다.
- Contour는 객체의 외곽선을 분석한다.
    

④ 실습과제

1. 네 가지 Blur 기법의 결과를 비교한다.
2. Sobel과 Canny의 차이를 비교한다.
3. Erosion과 Dilation을 구현한다.
4. Contour를 이용해 이미지 속 도형의 개수를 구한다.
5. Hough Transform으로 도로의 차선을 검출한다.
    

## 4단계. 이미지 분석 및 동영상 처리

Computer Vision 응용 : 권장 2일

[Online Image Histogram Viewer at Rene Neal blog](https://images.openai.com/static-rsc-4/drLnrJa4xoBS1V9QRuXh1g2ZxUjQPeRfQ6PkgUO2EpXgB8j7JFJgHMuxt0ST_DlQ2blZWaG9YiEuZsiBHqZFlKjeZZupLtQ0DGfKEKrf-LRRNnjMPUaTOBkZj9OH_K97NVp2CRWTT7H4tYT9QKQv7cNmIqlFy848NWass_x3oS0?purpose=inline)

[[Python OpenCV] OpenCV-python을 활용한 객체추적 프로그램. | flowerroot](https://images.openai.com/static-rsc-4/k4Cask3310Dop9LAkQIZsxiwfdIuUQaCx9apxKLO6BG_YKoAlCYBYAM--1hyX11Zr036IELX6QhDmNhHc-Crw8frofhwzSzsgoCC8AsHXde2GB0N0Ok2lhKA70xWtE2oLzmYnl9BjpHfEzPDaxuznCGFimMSWMKnJxy7_GqCoSg?purpose=inline)

[Moving Vehicle Detection – Vision & Graphics Group](https://images.openai.com/static-rsc-4/N39ekK149ncHLSYosCbFxXR2l0alVvuXhwiyvQF7lyCPJYarmMEQiHnSFScp9O2jzxZy216LEJJzsrIImyxX_uuKfGU8lW4fbG-a5Tuo-rvheS3Qg1x-zfA5jISSN66eTH8UG5rqIhFnA0PpB0jLbURASFvy_ePXBxNytIfLZK0?purpose=inline)


① 수업목표

- 이미지의 밝기 및 명암 분포를 분석한다.
- 관심 영역을 지정하여 특정 객체를 분석한다.
- 동영상을 프레임 단위로 처리하는 방법을 익힌다.
    

② 수업내용

|세부 주제|학습 내용|
|---|---|
|Histogram|이미지 밝기 분포|
|Histogram Equalization|명암 대비 개선|
|CLAHE|지역적 명암 개선|
|ROI|관심 영역 추출|
|Background Subtraction|배경과 전경 분리|
|Optical Flow|객체 움직임 분석|
|Object Tracking|객체 추적|
|Video Processing|동영상 프레임 처리|
|FPS|영상 처리 속도 측정|

주요 함수:

```python
cv2.calcHist()
cv2.equalizeHist()
cv2.createCLAHE()
cv2.VideoCapture()
cv2.VideoWriter()
cv2.calcOpticalFlowPyrLK()
```

③ 핵심 요약

- 이미지는 정적인 데이터이고 동영상은 연속적인 이미지 프레임이다.
- ROI는 전체 이미지 중 분석이 필요한 영역이다.
- Tracking은 프레임 간 객체의 위치 변화를 추적한다.
- FPS는 초당 처리하는 프레임 수를 의미한다.

④ 실습과제

1. Histogram을 이용해 이미지 밝기 분포를 분석한다.
2. Equalization과 CLAHE를 비교한다.
3. 웹캠 또는 동영상에서 ROI를 설정한다.
4. Optical Flow를 이용해 움직임을 검출한다.
5. 동영상을 읽어 처리한 후 새로운 동영상으로 저장한다.

# 3. 딥러닝 기반 Computer Vision

여기부터는 OpenCV만으로 구현하던 전통적인 영상 처리에서 PyTorch와 torchvision을 활용하는 딥러닝으로 넘어갑니다.

#### 5단계. CNN(Convolutional Neural Network)

Deep Learning : 권장 3일

[What Is a Convolutional Neural Network? - IABAC](https://images.openai.com/static-rsc-4/RhtxbRXxmtyQX9Nm8rOhhktVh9I4pshd2iYHkNK-_rDvOXBtI85UKyr0c6DGnky4M-hm4Ks98GtcN9TnEOaa_P6-JUxsmnTWe_ODixbBZV2fQwNXBw0E0I_xMynObTTu8f7n86yWDQQB7kYVbzIu-yfkjtKJBzGeHOSCHOtobyI?purpose=inline)

[How will channels (RGB) effect convolutional neural network? | ResearchGate](https://images.openai.com/static-rsc-4/T7NtsCcxmRlnpmEMHUqCZ4E8QJEwNenv1aeHForGQU7CGZtrMqKsfotRfVUcY_LQVS-sGP4pFkmasd6LUgndwPy8AHG3qQxA1O2DH_1EEWpSivZ_aWqSAgY5lHIohfW3Bg-r5kubXDMsqbIzMm-q7c8IlXnZpvqekmnsxNW1ajw?purpose=inline)

[Neural Network Diagram Generator — CNN Visualizer | SciFig](https://images.openai.com/static-rsc-4/8c9o1C12BKh663SDqV2OD9nR0eY5f3F0ajhjRKpdKJLmzkfBiavwJtUOiVPbeoUVfM5W0cZXFGBciBg5k6clcZL4B-1eM-kqEDe9myZG1gFdwENRGd3eWHqUer6zN4cw-uN5UBk0BN_wMjuuMRXNZvsBJ_bymyCljz0sI0ajk8M?purpose=inline)


① 수업목표

- CNN의 구조와 동작 원리를 이해한다.
- 이미지의 특징을 자동으로 추출하는 과정을 이해한다.
- PyTorch를 이용해 이미지 분류 모델을 구현한다.
    

② 수업내용

|세부 주제|학습 내용|
|---|---|
|ANN 복습|Neural Network 구조|
|Convolution|합성곱 연산|
|Filter / Kernel|특징 추출 필터|
|Feature Map|특징 맵|
|Stride / Padding|출력 크기 조절|
|Pooling|특징 축소|
|Activation|ReLU, Softmax|
|Flatten|다차원 데이터 변환|
|Fully Connected|최종 분류|
|Loss Function|Cross Entropy|
|Optimizer|SGD, Adam|
|학습 과정|Forward / Backward|

③ 핵심 요약

CNN은 이미지에서 특징을 자동으로 학습하는 신경망입니다.
![[Pasted image 20261110151943.png]]

- Convolution: 이미지 특징 추출
- Pooling: 특징 맵의 크기 축소
- Fully Connected: 최종 분류
- Backpropagation: 오차를 이용한 가중치 업데이트

④ 실습과제

1. NumPy로 간단한 Convolution 연산을 구현한다.
2. PyTorch로 CNN 모델을 구현한다.
3. MNIST 데이터셋으로 숫자 분류 모델을 학습한다.
4. CIFAR-10 데이터셋으로 이미지 분류를 수행한다.
5. Epoch, Batch Size, Learning Rate에 따른 정확도를 비교한다.
    

## 6단계. 전이학습(Transfer Learning)

Deep Learning 응용: 권장 2일

[Comparison of Deep Neural Networks in Detecting Field Grapevine Diseases Using Transfer Learning | MDPI](https://images.openai.com/static-rsc-4/iG7RmsvzwLWlvozhzkeiG70OaOY1EWqpnQElAQUhp3AFV9PFHkvdwmxdDdZylc3kq82z3LWMyfb1U_S0GWt8fbmjbA8t3htUl8a7leSxka-9q1U37rTLyuIrCpcELWqAVPkm_ReVKRgQqKcqgJGPLBckTcwCfGM3a-roMlQSbLs?purpose=inline)

[Modern Convolutional Neural Network Architectures](https://images.openai.com/static-rsc-4/VExfU9ptQS_TBcMqDUmNt5g-57aGMbXeDAunVQems4m2hT_RRGZReGKtDZntj6NNKRj3MuFXSPRs1Ovr0RhLtGWoDsKOABmPYf9FosGyYjTtsG6PX9uS9dv8QyBgBPl8f_37cfH3u3cl--tnN7bb_8LwudGo6m6f6ugS6tRYpis?purpose=inline)

[Review: MobileNetV2 — Light Weight Model (Image Classification) | by Sik-Ho Tsang | TDS Archive | Medium](https://images.openai.com/static-rsc-4/jrHQ-lyKWt9vKc-lu4_4vSOAeca7HJQqohFue6wVZl3XKNsvpqqzGqI_qMUTEwKx92Mgg9D1QTLUDd6s2z5jytl33tXmg0Vmkm6J7Aup8Hmp8-W_u-Y9Un-BmWRY9VvTwg5d2CqM3GP1f-pAB_ij_hvVHOAVQZfIBVvOSXYA2i8?purpose=inline)


① 수업목표

- 사전학습 모델의 개념을 이해한다.
- 기존 CNN 모델을 새로운 데이터셋에 적용한다.
- Feature Extraction과 Fine-tuning의 차이를 이해한다.
    

② 수업내용

|세부 주제|학습 내용|
|---|---|
|Pretrained Model|사전학습 모델|
|Transfer Learning|전이학습|
|Feature Extraction|특징 추출기 활용|
|Fine-tuning|전체 또는 일부 재학습|
|ResNet|Residual Learning|
|MobileNet|경량 CNN|
|EfficientNet|효율적인 모델 설계|
|Data Augmentation|데이터 증강|
|Model Evaluation|Accuracy, Precision, Recall|

실습 모델:

- ResNet18
- MobileNetV2
- EfficientNet-B0

③ 핵심 요약

전이학습은 대규모 데이터셋으로 미리 학습한 모델의 지식을 새로운 문제에 활용하는 방법입니다.

|구분|Feature Extraction|Fine-tuning|
|---|---|---|
|기존 특징 추출층|고정|일부 또는 전체 학습|
|분류기|교체 및 학습|교체 및 학습|
|학습 시간|상대적으로 짧음|상대적으로 김|
|데이터 요구량|비교적 적음|상황에 따라 더 필요|

④ 실습과제

1. ResNet18을 이용해 이미지 분류를 구현한다.
2. 기존 분류기를 새로운 클래스에 맞게 교체한다.
3. Feature Extraction 방식으로 학습한다.
4. Fine-tuning 방식으로 학습한다.
5. Scratch 학습과 전이학습의 정확도 및 학습 시간을 비교한다.
    

## 7단계. 이미지 세그멘테이션(Image Segmentation)

Deep Learning 응용 : 권장 3일

[Examples – Cityscapes Dataset](https://images.openai.com/static-rsc-4/f7n9UOOU_Gs9eABprXyQKdu93oGWB8bPeIa1PhuBvGryOGR1Dcy8Uu3hIoOHhj2CVJjpYDL2pHHV5htBO_dtpVT6GOpqL4TF5aXqL9jyclkv_Bc9IQYpMkX-Va0lG4qFj_H224VmrKnmQhaol5oKqVQ9O8Jgn49fu1PcgJq3urA?purpose=inline)

[Mask RCNN －原理解析（一）_mask rcnn原理 to be better one-CSDN博客](https://images.openai.com/static-rsc-4/Hh9Z4o864RPLcj5B0UHhWwsyL316Y0TxANVQBsCeSouGbFNFl9CVGTDxWwYsWrUWuHL-6W3kpbDfYv8P3_YXpiCxeMiEUw9K8LFWoU9tEZ_yPhBtjz7hnZtEOhtaQ-lcvByb4QOB3QWJMQ5P1H6vEz5NvqdrMyREt8bUKFXhE6o?purpose=inline)

[Visualization of ground truth — Part-aware Panoptic Segmentation 2.0rc5 documentation](https://images.openai.com/static-rsc-4/3HG145NwwLRONxZfMadJGkjz8lJz43x8RyFyA_aG7httOJbK0yIZjYsnspmVguNG4P2bRKZW0t7DPiexELJ8Gtt50kkrpHLL2ZrDC_xqM6ik2XC_-Ae3E05aY4f5zNgGFEUwXHdBHQAD4y0nxCZPnnhn0Litc7jDo3hZ7bXlPx4?purpose=inline)


① 수업목표

- 이미지 분류, 객체 검출, 세그멘테이션의 차이를 이해한다.
- 픽셀 단위의 이미지 분할 원리를 이해한다.
- 딥러닝 모델을 이용해 객체 영역을 분할한다.

② 수업내용

|세부 주제|학습 내용|
|---|---|
|Segmentation 개념|Pixel-wise Classification|
|Semantic Segmentation|클래스 단위 분할|
|Instance Segmentation|객체 개별 분할|
|Panoptic Segmentation|Semantic + Instance|
|FCN|Fully Convolutional Network|
|U-Net|Encoder-Decoder|
|DeepLabV3|Atrous Convolution|
|Mask R-CNN|객체별 마스크 생성|
|평가 지표|PA, IoU, mIoU, Dice|

③ 핵심 요약

세그멘테이션은 이미지의 각 픽셀이 어떤 클래스에 속하는지를 판단하는 기술입니다.

|기술|분석 단위|결과|
|---|---|---|
|Classification|이미지 전체|고양이|
|Object Detection|객체 영역|고양이 + Bounding Box|
|Semantic Segmentation|픽셀|고양이 영역|
|Instance Segmentation|개별 객체 픽셀|고양이 1, 고양이 2|

④ 실습과제

1. Semantic Segmentation의 결과 마스크를 시각화한다.
2. U-Net으로 간단한 세그멘테이션을 구현한다.
3. DeepLabV3를 이용해 도로와 하늘을 분할한다.
4. Mask R-CNN으로 객체별 마스크를 추출한다.
5. PA, IoU, mIoU, Dice Score를 계산하고 비교한다.

## 8단계. Vision Transformer

최신 Computer Vision : 권장 2~3일

[Train Vision Transformer model and run Inference | Microsoft Community Hub](https://images.openai.com/static-rsc-4/Nt7sbbg8ByQTzGAtZj9mArklQRynBq5sYOp4e6kzUg10leQxoqFxmPXEOjoqpugKOzDfKAyKolbzv9lqqPH50HFPfcF6CbxMkzdyOLc3RvX2EY0BKnUTs63c4bd1jp80CB5Wy08XcHoZsSlZRPe0IkS55mB3iR2m55SeTFvgL5c?purpose=inline)

[What is Transformers in Deep Learning: A Guide 2026](https://images.openai.com/static-rsc-4/xO4fPox9ELTrYHdqurOpfbPtoHytHgEgmh5g8rkWfhUFfpkH9hgzv4zmeWEnGHh6LGEIkw3D6VcTWPZ3RiFzyq_UH16e76U4NB1nyfR4-DpZQ3PFTFCPpFlP8Y0-0F4Qn-xpySScL0lnZIl-65d5pOF-xekJxjxZSWjZYNm4X1o?purpose=inline)

[DETR: End-to-End Object Detection with Transformers(ECCV 2020)](https://images.openai.com/static-rsc-4/3jGh0j5gUhBBUUY31XhrzFLDZCA4bXCLyuLiq3AM_g_o-pQCahEz7EtkP4JazWZbHshVhJx-UcV1Almlo4coIw1ZUFtQ88M0f4vC4KR0t0RIjyh7h3_S6zH-PkMKlRxLXHPlP-OUvyr3YQPImDZjbHvAzz-sYshVrsR4YWm6ukA?purpose=inline)

① 수업목표

- Transformer의 기본 구조와 Attention 개념을 이해한다.
- 이미지에 Transformer를 적용하는 원리를 이해한다.
- CNN과 Vision Transformer의 차이를 설명할 수 있다.

② 수업내용

|세부 주제|학습 내용|
|---|---|
|Transformer|Transformer 기본 구조|
|Attention|Attention Mechanism|
|Self-Attention|자기 어텐션|
|Multi-Head Attention|다중 헤드 어텐션|
|Positional Encoding|위치 정보|
|Patch Embedding|이미지를 패치로 분할|
|Vision Transformer|ViT 구조|
|CNN vs ViT|특징 추출 방식 비교|
|DETR|Transformer 기반 객체 검출|

③ 핵심 요약

CNN은 커널을 이용해 이미지의 지역적인 특징을 추출하는 반면, Vision Transformer는 이미지를 여러 개의 패치로 나누고 Self-Attention을 통해 패치 사이의 관계를 학습합니다.

CNN과 ViT 비교

[Pooling layer - Introduction to Deep Learning - Medium](https://images.openai.com/static-rsc-4/n_F2OtQTsXF0xvh_prqs1BoZZfNy_AWtX47eaRuSA1SAvwYsU9mYoUjyZJA3LphFGFdEipEhs9J1JGnkb6Q_WCvxYWMKFcwRmn39cCLjx_Li6IqtknWiM0dnerY6NCy9HIyakCPYRGNK1h28EJKOnTi6fuZjafT15kYJAVxgQiQ?purpose=inline)

CNN

Convolution 기반 지역 특징 추출

[Vision Transformer (ViT) Architecture Showdown | by Saqib Khan | Medium](https://images.openai.com/static-rsc-4/39poFuCjNM1H3Q6UDRyi33qpLecNnzwZEcj3RELY-cww-5VFryi-s7jJ6Ymu_X2paZmwIO_9OjVeklsRCWY2zIThh_RBvfiEh3_5TXGM-Qi3NLglTC8Q2Qd0xqoLM36-n7amI4WUgNxNRcLNSqDs3wV9TEpLVYXdryhuvlQEqR8?purpose=inline)

ViT

Patch + Self-Attention 기반 특징 학습

주의할 점은 Transformer가 CNN을 완전히 대체하는 것은 아니라는 것입니다. 데이터 크기, 모델 구조, 연산 자원에 따라 장단점이 다릅니다.

④ 실습과제

1. Transformer의 Self-Attention 동작 원리를 확인한다.
2. 이미지를 Patch 단위로 분할한다.
3. 사전학습된 ViT로 이미지 분류를 수행한다.
4. ResNet18과 ViT의 분류 성능을 비교한다.
5. DETR의 Bounding Box 예측 결과를 시각화한다.

## 9단계. YOLO 객체 검출

실전 Computer Vision : 권장 3~4일

[YOLO: The AI Model Powering Real-Time Object Detection | by Wicar Akhtar | Medium](https://images.openai.com/static-rsc-4/Fp_LqGz1v18i76gYTQHjqMDn8Hz010_vhtFlXLCvQfkIgpq12tCJ6ny01H837Hmd7Sk4f3rM2AVDclpMTF8QMlewTj0OX7g5L02jr8F5fsDCDDdD0sAOcpaDwfw_nUtD7iB5DT-BKAWg3qZh2XtIIMixvICbuXgBM-uFULNgsFs?purpose=inline)

[Enhancing YOLOv8’s Performance in Complex Traffic Scenarios: Optimization Design for Handling Long-Distance Dependencies and Complex Feature Relationships | MDPI](https://images.openai.com/static-rsc-4/QhmMGcai08Fe_KP27uhdV5J-FR0PRuwsR5oUdFCkvwYvrz0yvNlh0NyQGDDlmXqY58Gp8WZ0Idr_sWcII69VYc_WiOs7YupTHPZOB8HIv1nlX9Qzi9JeeObCTjgGkbXg0aQqFsRxBr6J-tvn3JJTcqWy_kyKk2lgVgUXGOAlKmQ?purpose=inline)

[The Best Object Detection Methods for 2023 | A Comprehensive Guide](https://images.openai.com/static-rsc-4/b2UyEzVZkWWBSqezq7h9D_vW3tVDAqDKMD8lTAmAWTR24pZrD7JIJy_JDz1Hd_g5VN8LTNFbYagkXI24jVohhaebTg04MornUJYdCXuwWIA7LLHSbP12oxW8m9SmQt3K5O1mamG94eg0kChJyMMszT6FYACw1lHsUNgXhQ5hOm8?purpose=inline)


① 수업목표

- 객체 검출의 기본 구조를 이해한다.
- YOLO의 모델 구조와 객체 검출 과정을 이해한다.
- 사전학습 모델을 이용한 객체 검출과 커스텀 학습을 수행한다.
    

② 수업내용

|세부 주제|학습 내용|
|---|---|
|Object Detection|객체 검출 개념|
|Bounding Box|객체 위치 표현|
|IoU|박스 중첩 정도|
|Confidence|검출 신뢰도|
|NMS|중복 박스 제거|
|YOLO Architecture|Backbone, Neck, Head|
|YOLO 모델|YOLO 계열 모델 발전|
|Pretrained YOLO|사전학습 모델|
|Custom Dataset|사용자 데이터셋|
|Annotation|객체 라벨링|
|Training|모델 학습|
|Evaluation|Precision, Recall, mAP|
|Deployment|ONNX, TensorRT 등|

실습 모델은 현재 사용 가능한 Ultralytics YOLO 모델을 중심으로 진행하는 것을 추천합니다.

③ 핵심 요약

YOLO는 이미지에서 객체의 종류와 위치를 동시에 예측하는 객체 검출 모델입니다.

- Backbone: 이미지 특징 추출
- Neck: 다양한 크기의 특징 결합
- Head: 객체의 클래스와 위치 예측
- NMS: 중복 검출 결과 제거
- mAP: 객체 검출 성능 평가
    

④ 실습과제

1. 사전학습 YOLO 모델로 이미지 객체를 검출한다.
2. 동영상에서 자동차, 사람, 자전거를 검출한다.
3. 신호등 및 교통 표지판 검출을 수행한다.
4. 직접 만든 데이터셋으로 YOLO를 학습한다.
5. YOLO 모델의 Precision, Recall, mAP를 평가한다.
6. ONNX로 모델을 변환하고 추론 결과를 비교한다.
    

# 4. 종합 프로젝트

마지막 단계에서는 앞에서 학습한 OpenCV와 딥러닝 기술을 통합하는 프로젝트를 진행합니다.

최종 프로젝트

## 프로젝트 A. 자율주행 영상 분석 시스템

[Udacity Self-Driving Car Engineer Nanodegree Projects | by David Silver | Udacity Inc | Medium](https://images.openai.com/static-rsc-4/zQ86ehRApF4Sr0JVE7iDgFVY9jTcF5X0iB6OYXSvlzNKylNQtwHJDU33amt1w2NFnyZ6wIZiweKQGlIUrZeKnTAkfEU2K7X57fausrhXIHM0bKBvv6kzWcMpa_1hsjO_wz1XGEpHhYmoOxzjuU3HSqcADRAEWeVXD285eCvKWBM?purpose=inline)

[What is Computer Vision (CV)? | BasicAI Wiki](https://images.openai.com/static-rsc-4/DPNr9Bnm3bxuJwI4XQfK-4cePHAMs3j8mEinlrGcwe_5hdXcxn0Jt2Rego7e9MBwPVH_Ecenb6YPTTPDoythAoQFpnG806O0ME7tfQGjT5sB5978oxbZ34DRRRm4OZoQvwhmFp155Id3tU3wk8Z22dL_2SyQe3fFH4RDEr8tj04?purpose=inline)

[Sichere autonome Autos: Branche legt Grundstein für ISO-Norm](https://images.openai.com/static-rsc-4/JbV_KfHDHWu5XtiCTEkpOKieNin2V8quz_3jx0icAHhyxwRbGOG39ZjphAMdghfaj3v8LQ4wI3rnHXDCYmuPkhcoWF9-bl6_ZwfQyFvtbOHJyFutbjAGaJ0u2FNFyNOGdYGOgpiO5cffCSin38haLe2pH4Rzdv0IBFLPhsHZ5OU?purpose=inline)


프로젝트 목표

블랙박스 동영상에서 차선, 자동차, 보행자, 신호등 등을 검출하고 주행 상황을 분석하는 시스템을 구현합니다.

|모듈|적용 기술|
|---|---|
|이미지 전처리|OpenCV|
|차선 검출|Canny, Hough|
|객체 검출|YOLO|
|도로 영역 분할|DeepLabV3|
|객체 추적|Tracking|
|주행 판단|위치 및 움직임 분석|
|결과 시각화|OpenCV|
|성능 평가|FPS, Latency|

실습과제

- 차선 검출 및 차선 이탈 판단
- 전방 차량과 보행자 검출
- 신호등 상태 분석
- 횡단보도 및 정지선 검출
- 객체 추적 및 주행 상황 판단
- 분석 결과 영상 및 CSV 저장

## 프로젝트 B. 실시간 객체 인식 시스템

[Object Detection from Webcams with YOLO using Python - wellsr.com](https://images.openai.com/static-rsc-4/BBUth1-GkolKtbLJKnuVMymNt8o6BC1FdJsuxlZ5MBihrJxKEoOvkl0JVbAfSRBbzNck_j0IZ3LfMYg6LmKbQnnnoQGmzBfcI48ooqkHnVg72yBIrJRqpGNIcTP9HuaFRyMn-se_4Zl7NrsKKhP9FuBGDdv6YdDDN4rVc2n2LwE?purpose=inline)

[Covid-19 Alert System - Jetson Projects - NVIDIA Developer Forums](https://images.openai.com/static-rsc-4/f_gO4VQjOZqt6BibqJAfUoClHQNABEubLf_7B_S-ypaT8608xqiL8_Xo3HyqbbeSKHKi5ggCmYy2B2qmENMmUlwSZlGoIKtE44Gnju7D-4vCBr2KVL6AfE-50I4Qzxk24HJGhS8y2TS5T2gP0go64sSans8_7Uk4Hv-Way6tJDk?purpose=inline)

[Running YOLO on NVIDIA Jetson Orin Nano](https://images.openai.com/static-rsc-4/l1gltGeRvYu5ZGuGwBYgK_Nfp68geL79Jaj8JoN5axHQYhSycPFRwdnFjBRQtvGYcj_24rEotk29fNpfJQQCBotHQPLcA9nwGjLcKzxLCtlzvx68ocXDHk5CkkyrAIu54GDvBqrl3wMa_DN3uU9qDjS1a33QpAk7QjRvBWDwU_U?purpose=inline)


프로젝트 목표

웹캠 또는 카메라 영상을 입력받아 실시간 객체 검출 및 추적을 수행합니다.

- OpenCV VideoCapture
- YOLO 객체 검출
- 객체 Tracking
- FPS 측정
- 모델 경량화
- ONNX 추론
- Jetson Nano 등 임베디드 환경 배포
    

# 5. 전체 교육 평가 및 산출물

각 단계에서 단순히 이론을 이해하는 것에 그치지 않고 결과물을 남기는 것이 중요합니다.

|단계|평가 항목|산출물|
|---|---|---|
|1|이미지 데이터 구조 이해|이미지 분석 코드|
|2|이미지 변환 구현|전처리 결과|
|3|필터링 및 특징 추출|차선 및 도형 검출|
|4|영상 분석|동영상 처리 프로그램|
|5|CNN 모델 구현|이미지 분류 모델|
|6|전이학습|학습된 분류 모델|
|7|Segmentation|픽셀 분할 결과|
|8|Transformer|ViT 분류 결과|
|9|YOLO|객체 검출 모델|
|10|통합 프로젝트|영상 분석 시스템|

## 6. 수업 진행 시 추천 개발 환경

|구분|추천 도구|
|---|---|
|언어|Python 3.11|
|이미지 처리|OpenCV|
|배열 처리|NumPy|
|시각화|Matplotlib|
|딥러닝|PyTorch|
|비전 모델|torchvision|
|데이터 증강|Albumentations|
|객체 검출|Ultralytics YOLO|
|데이터셋|MNIST, CIFAR-10, VOC, COCO|
|개발 환경|VS Code, Jupyter, Colab|

# 7. 강의 구성에서 중요한 교육적 포인트

제가 추천하는 핵심은 다음과 같이 세 가지 영역을 명확하게 구분해서 가르치는 것입니다.
![[Pasted image 20261110152413.png]]

추가로 교육할 때 반드시 강조할 부분이 있습니다.

1. OpenCV와 CNN은 경쟁 관계가 아닙니다. OpenCV는 이미지 입출력과 전처리 및 시각화에 계속 활용됩니다.
2. 전이학습은 CNN 이후에 배치하는 것이 이해하기 쉽습니다.
3. Segmentation은 CNN의 특징 추출 구조를 이해한 후 학습해야 합니다.
4. Transformer는 CNN과 비교하면서 학습하면 이해가 빠릅니다.
5. YOLO는 객체 검출의 개념(Bounding Box, IoU, NMS)을 먼저 학습한 뒤 진행해야 합니다.
6. 마지막에는 ONNX와 TensorRT를 이용한 경량화 및 임베디드 배포까지 연결하면 실무 교육의 완성도가 높아집니다.
    

최종 추천: 수업 자료는 각 단계마다 `수업목표 → 이론 요약 → 핵심 코드 → 단계별 실습 → 실습과제 → 평가문제` 형식으로 제작하는 것이 좋습니다. 특히 비전 기반 자율주행 교육과 연결한다면 OpenCV의 차선 검출에서 시작해서 YOLO 객체 검출, Segmentation, 최종 주행 판단까지 하나의 프로젝트로 연결할 수 있습니다.