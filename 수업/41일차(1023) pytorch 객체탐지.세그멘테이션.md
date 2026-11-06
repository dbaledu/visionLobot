
수업자료:  R-CNN.pdf,  YOLOY.pdf
참조 : 교재 4장 참조


## 수업목표
- pytorch 기반 객체탐지, 세그멘테이션
- 

## 실습파일
- ㅇㅇ
- 

이미지 처리
- 분류
- 탐지 = 분류(classification) +  region proposal(위치)
- 분할

탐지 모델중에서 YOLO가 제일 빠름
R-CNN 계열과 YOLO 계열로 발전. 객체탐지는 YOLO, 세그멘테이션은 R-CNN 이용
YOLO 1스테이지, R-CNN는 2스테이지

## R-CNN
```
seiective search -> crop & resize -> CNN -> cache -> Regressor(회귀)
                                                     SVMs
```

- R-CNN : 속도가 느림
- Fast R-CNN : CNN 처리가 앞에
- Faster R-CNN : 

mAP(mean Average Precision): 평가지표

#### YOLO (You Only Look Once)
- 2K(2000)개 클래스 탐지
- 4버전 - 다크넷 -> 경량화
- 임베디드(잭슨나노)
- 얼트럴리틱스([Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/?utm_source=chatgpt.com))

#### 1 stage / 2stage
- 2-Stage Detector - Regional Proposal와 Classificaton이 순차적으로 이루어짐
	- R-CNN
- 1-Stage Detector • Regional Proposal와 CIassificaoon이 동시에 이루어짐.
	- YOLO


## Darknet 실행 실습

###### Darknet 명령어 실행
```sh
/content/darknet# ./darknet detector test \
    cfg/coco.data \
    cfg/yolov4-tiny.cfg \
    yolov4-tiny.weights \
    data/dog.jpg \
    -thresh 0.25
```

이 명령은 다음 순서로 동작합니다.
1. `darknet` 실행 파일을 실행합니다.
2. `.cfg` 파일에서 네트워크 구조를 읽습니다.
3. `.weights` 파일에서 학습된 가중치를 읽습니다.
4. 이미지를 입력받아 객체를 탐지합니다.
5. 클래스 이름과 바운딩 박스를 결과로 출력합니다.

###### 실행결과
```bash
 CUDA-version: 13000 (13000), cuDNN: 9.14.0, GPU count: 1  
 OpenCV version: 4.6.0
 0 : compute_capability = 750, cudnn_half = 0, GPU: Tesla T4 
net.optimized_memory = 0 
mini_batch = 1, batch = 1, time_steps = 1, train = 0 
   layer   filters  size/strd(dil)      input                output
   0 Create CUDA-stream - 0 
 Create cudnn-handle 0 
conv     32       3 x 3/ 2    416 x 416 x   3 ->  208 x 208 x  32 0.075 BF
   1 conv     64       3 x 3/ 2    208 x 208 x  32 ->  104 x 104 x  64 0.399 BF
   2 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
   3 route  2                                  1/2 ->  104 x 104 x  32 
   4 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
   5 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
   6 route  5 4                                    ->  104 x 104 x  64 
   7 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
   8 route  2 7                                    ->  104 x 104 x 128 
   9 max                2x 2/ 2    104 x 104 x 128 ->   52 x  52 x 128 0.001 BF
  10 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
  11 route  10                                 1/2 ->   52 x  52 x  64 
  12 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
  13 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
  14 route  13 12                                  ->   52 x  52 x 128 
  15 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
  16 route  10 15                                  ->   52 x  52 x 256 
  17 max                2x 2/ 2     52 x  52 x 256 ->   26 x  26 x 256 0.001 BF
  18 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
  19 route  18                                 1/2 ->   26 x  26 x 128 
  20 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
  21 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
  22 route  21 20                                  ->   26 x  26 x 256 
  23 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
  24 route  18 23                                  ->   26 x  26 x 512 
  25 max                2x 2/ 2     26 x  26 x 512 ->   13 x  13 x 512 0.000 BF
  26 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
  27 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
  28 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
  29 conv    255       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 255 0.044 BF
  30 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
nms_kind: greedynms (1), beta = 0.600000 
  31 route  27                                     ->   13 x  13 x 256 
  32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
  33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
  34 route  33 23                                  ->   26 x  26 x 384 
  35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
  36 conv    255       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 255 0.088 BF
  37 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
nms_kind: greedynms (1), beta = 0.600000 
Total BFLOPS 6.910 
avg_outputs = 310203 
 Allocate additional workspace_size = 26.22 MB 
Loading weights from yolov4-tiny.weights...
 seen 64, trained: 0 K-images (0 Kilo-batches_64) 
Done! Loaded 38 layers from weights-file 
 Detection layer: 30 - type = 28 
 Detection layer: 37 - type = 28 
data/dog.jpg: Predicted in 52.940000 milli-seconds.
bicycle: 60%
dog: 84%
truck: 79%
car: 45%
qt.qpa.xcb: could not connect to display 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, xcb.

Aborted (core dumped)
```



