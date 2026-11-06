


### 수업목표
- 프로젝트: 재활용품을 분리하는 딥러닝 모델을 훈련한 후 활용할 수 있게 코렙과 윈도우에서 개발
- From scratch와 전이학습 모델 사용하여 영상과 이미지 추론하기 (교재 5장 참조)
- 

### 실습파일
- Inference_Cam.py
- 

### Inference_Cam.py

동영상을 입력받아 재활용품 추론하기

```python
# filename : Inference_Cam.py
# description : 동영상을 입력받아 재활용품 추론하기
# author : kym
# date : 2026/11/03

import os
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import numpy as np 
from Model_Class_From_The_Scratch import Model_From_Scratch    # 사용자 정의 학습
from Model_Class_Transfer_Learning_MobilNet import MobileNet   # MobileNet 전이 학습
import argparse   # 명령행 인수(실행파일 매개변수) 파싱

# 현재 작업중인 파일이 위치한 폴더의 절대 경로  __file__은 작업중인 파일명
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print('dir:',SCRIPT_DIR)
print('file 절대경로명:',__file__)

class Inference_Class():
    def __init__(self):
        USE_CUDA = torch.cuda.is_available()
        self.DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
        print('USE_CUDA', USE_CUDA)
        self.model = None
        self.label_map = None   # 클래스명
        #학습에 사용된 동일한 mean, std 적용
        self.trasnform_info = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(              # 이미지넷 평균/표준편차 정규화 
                mean=[0.485, 0.456, 0.406],    # ImageNet 데이터셋의 RGB 채널별 통계값으로 이미 정해져 있음
                std=[0.229, 0.224, 0.225]
            )
        ])
    #-----------------------------------------------------
    # 라벨맵 파일과 모델 가중치 파일 불러오기
    #-----------------------------------------------------    
    def load_model(self, is_train_from_scratch, label_map_file="label_map.txt"):
        '''
        라벨맵 파일과 모델 가중치 파일 불러오기

        Args
            is_train_from_scratch: True 이면 전이학습 신경망과 가중치를 불러옴.
                                   False 이면 직접 구현해 훈련했던 신경망과 가중치를 불러옴
            label_map_file: 분류할 클래스 이름
        '''      

        # 상대경로로 라벨 파일을 찾아보고 없으면 절대경로로 찾음
        if not os.path.exists(label_map_file) and not os.path.isabs(label_map_file):
            candiate = os.path.join(SCRIPT_DIR, label_map_file)
            if os.path.exists(candiate):
                label_map_file = candiate
        print('lable_map', label_map_file)
        self.label_map = np.loadtxt(label_map_file, str, delimiter="\t")

        num_classes = len(self.label_map)             # 클래스 개수
        model_str = None                              # 가중치 파일명

        # 모델과 학습한 가중치 파일 설정
        if is_train_from_scratch:
            self.model = Model_From_Scratch(num_classes).to(self.DEVICE)
            model_str = 'Pytorch_Train_From_Scratch'
        else:
            self.model = MobileNet(num_classes).to(self.DEVICE)
            model_str = 'Pytorch_Transfer_Learning_Mobilenet'
        model_str += '.pt'

        # label_map_file 파일경로 처리 방식과 동일
        if not os.path.exists(model_str) and not os.path.isabs(model_str):
            candiate = os.path.join(SCRIPT_DIR, model_str)
            if os.path.exists(candiate):
                model_str = candiate
        
        # 가중치를 모델에 적용하고 GPU에서 학습한 가중치를 CPU에서도 불러올 수 있도록 지정
        self.model.load_state_dict(torch.load(model_str, map_location=self.DEVICE))

        # 평가모드 전환
        self.model.eval()

    #-----------------------------------------------------
    # 동영상 파일, 카메라 영상을 프레임 단위로 읽어 실시간 추론
    #-----------------------------------------------------
    def inference_video(self, video_source='test_video.mp4'):
        '''
        동영상 파일, 카메라 영상을 프레임 단위로 읽어 실시간 추론

        Args
            video_source : 입력 비디오 파일명

        '''      
        
        # 문자열 정보인지 체크
        if isinstance(video_source, str):
            # 문자열 정보 '0', '1' 이면 카메라 장치번호 숫자로 변환 -> 일반 USB 카메라(웹캠)
            if video_source.isdigit():
                video_source = int(video_source)
            # 현재 폴데에 동영상이 없으면 절대 경로로 이용해서 탐색
            elif not os.path.exists(video_source) and not os.path.isabs(video_source):
                candiate = os.path.normpath(os.path.join(SCRIPT_DIR, video_source))
                if os.path.exists(candiate):
                    video_source = candiate
        # OpenCV로 영상 열기
        cap = cv2.VideoCapture(video_source)

        if cap.isOpened():
            print('video opened')
        else:
            print('vido not opened')
            print('program abort')
            exit()
        
        cv2.namedWindow('Output', cv2.WINDOW_GUI_EXPANDED)

        with torch.no_grad():
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    output = self.inference_frame(frame)
                    cv2.imshow("Output", output)
                else:
                    break
                # 33ms 딜레이(초당 30프레임 정도) 및 'q' 입력시에 종료
                if cv2.waitKey(33) & 0XFF == ord('q'):
                    break
            # 리소스 해제 및 모든 창 닫기
            cap.release()
            cv2.destroyAllWindows()
        return

    #-----------------------------------------------------
    # 프레임 한장을 분류하고 예측
    #-----------------------------------------------------
    def inference_frame(self, opencv_frame):
        '''
        프레임 한장을 분류하고 예측

        Args
            opencv_frame: 입력 프레임

        Returns
            result_frame: 예측결과(라벨을)를 표시한 프레임
        '''

        # frame 이미지 전처리

        opencv_rgb = cv2.cvtColor(opencv_frame, cv2.COLOR_BGR2RGB)
        # numpy -> PIL 이미지
        image = Image.fromarray(opencv_rgb)
        # 전처리 적용하여 텐서로 변환 (h,w,c)  -> (c,h,w)
        image_tensor = self.trasnform_info(image)
        # 배치 차원 추가
        image_tensor = image_tensor.unsqueeze(0)
        # 모델과 같은 연산 장치로 이동
        image_tensor = image_tensor.to(self.DEVICE)

        # 추론 적용
        inference_result = self.model(image_tensor)
        # 배치 차원 제거
        inference_result = inference_result.squeeze()
        # GPU - > CPU 이동후 numpy array 변환
        inference_result = inference_result.cpu().numpy()
        # 원본 복사
        result_frame = np.copy(opencv_frame)
        # 확률이 가장 높은 argmax값으로 클래스 이름 조회
        label_text = self.label_map[np.argmax(inference_result)]
        # 클래스명 뒤에 확률값 추가
        label_text += ' ' + str(inference_result[np.argmax(inference_result)])

        result_frame = cv2.putText(
            result_frame, 
            label_text, 
            (10,50), 
            cv2.FONT_HERSHEY_PLAIN, 
            fontScale=2.0,
            color=(0,0,255),
            thickness=3)
            
        return result_frame


    #-----------------------------------------------------
    # 단일 이미지 추론
    #-----------------------------------------------------
    def inference_image(self, opencv_image):   
        '''
        단일 이미지 추론

        Args
            opencv_image : 입력 이미지

        Returns
            result_frame : 예측라벨이 표시된 결과 이미지
            label_text : 예측 클래스명
            class_prob : 예측 확률값
        
        '''      
        
        # frame 이미지 전처리
        opencv_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(opencv_rgb)
        image_tensor = self.trasnform_info(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.DEVICE)

        # 추론
        # 기울기 계산 비활성화
        with torch.no_grad():
            inference_result = self.model(image_tensor)
        inference_result = inference_result.squeeze()
        inference_result = inference_result.cpu().numpy()

        result_frame = np.copy(opencv_image)
        label_text = self.label_map[np.argmax(inference_result)]
        label_text += ' ' + str(inference_result[np.argmax(inference_result)])
        class_prob = str(inference_result[np.argmax(inference_result)])
        result_frame = cv2.putText(
            result_frame, 
            label_text, 
            (10,50), 
            cv2.FONT_HERSHEY_PLAIN, 
            fontScale=2.0,
            color=(0,0,255),
            thickness=3)

        #         결과 이미지,  예측 클래스명, 예측 확률값    
        return result_frame, label_text, class_prob        

def image_prob():
    image_path = "C:\\vision_work\\1030CNN\\Recycle_Classification_Dataset\\can\\bandicam 2019-11-03 19-31-13-204.jpg"
    image = cv2.imread(image_path)
    ret, classname, accuracy = inferenceClass.inference_image(image)
    print('class accu: ', classname, 'accuracy', accuracy)
    cv2.imshow('image',ret)
    cv2.waitKey() 


#--------------------------------------------
# 실행파일 옵션 사용법
#--------------------------------------------
# python 파일명               : MobileNet 전이학습 + test_video.mp4
# python 파일명  -s           : 사용자 정의 학습 모델 사용 
# python 파일명  -src 0       : 0번 카메라 사용 
# python 파일명  -src 동영상   : 다른 동영상 사용
#--------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -s 옵션 사용: 사용자 정의 모델. 사용하지 않으면 MobileNet 전이 모델 사용
    parser.add_argument(
        "-s", "--is_scratch", 
        required=False, 
        action="store_true",                                    # 옵션을 입력하면 자동으로 True
        help ="Inference with model trained form the scratch"
    )

    # -src : 동영상 파일 또는 카메라 번호
    parser.add_argument(
        "-src", "--source", 
        required=False, 
        type=str, 
        default="./test_video.mp4",
        help="OpenCV Video source"
    )

    args = parser.parse_args()
    is_train_from_scratch = False
    # 실행파일에 전달되는 매개변수
    source = args.source

    if args.is_scratch:
        is_train_from_scratch = True
    
    if source.isdigit():
        source = int(source)
    # 추론 클래스 객체 생성
    inferenceClass = Inference_Class()
    inferenceClass.load_model(is_train_from_scratch)

    # 동영상 추론. 단일 이미지 사용시에는 inference_image() 사용
    inferenceClass.inference_video(source)

    # 이미지 추론
    #image_prob()
```

### 요약
```python
# -------------------------------------------------------
# 1. 라벨맵(Label Map) 파일 읽기
# -------------------------------------------------------
# 라벨맵: 학습할 때 사용했던 클래스 이름(정답 레이블)을 파일에서 읽어옴
self.label_map = np.loadtxt(label_map_file, str, delimiter="\t")

# -------------------------------------------------------
# 2. 학습된 모델 구조 생성
# -------------------------------------------------------
self.model = MobileNet(num_classes).to(self.DEVICE)

# -------------------------------------------------------
# 3. 학습된 가중치(Weight) 불러오기
# -------------------------------------------------------
# 학습된 파라미터(Weight, Bias)는 파일에서 가져오기
# load_state_dict(): 읽어온 가중치 값을 현재 모델의 각 레이어에 적용
self.model.load_state_dict(
    torch.load(
         model_str, 
         map_location=self.DEVICE
    )
)

# -------------------------------------------------------
# 4. 추론(Inference) 모드 설정
# -------------------------------------------------------
# 학습이 아닌 예측을 수행하는 모드로 전환
# Dropout 비활성화, BatchNorm의 학습 통계 업데이트 중지
self.model.eval()

# -------------------------------------------------------
# 5. 추론 실행
# -------------------------------------------------------
# torch.no_grad()
# : 추론 과정에서는 기울기(Gradient)를 계산할 필요가 없으므로
#   역전파를 위한 연산을 생략하여 메모리 사용량을 줄임
with torch.no_grad():
    # image_tensor를 모델에 입력
    # 모델의 forward()가 실행되어 각 클래스에 대한 예측값 반환
	inference_result = self.model(image_tensor)

# -------------------------------------------------------
# 6. 가장 높은 예측값의 클래스 인덱스 찾기
# -------------------------------------------------------
# 확률이 제일 높은 결과
predicted_index = np.argmax(inference_result)

#torch를 사용하는게 더 자연스러움
predicted_index = torch.argmax(
    inference_result,
    dim=1
).item()

# -------------------------------------------------------
# 7. 예측 인덱스를 실제 클래스 이름으로 변환
# -------------------------------------------------------
label_text = self.label_map[predicted_index]
```

### 전체 추론과정 파이프라인

![[1103.png]]
