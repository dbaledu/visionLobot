
## 수업목표
- CNN(tensorflow)
- 전이학습이란

## 실습파일
- cv1015_1.ipynb
	- mnist
- cv1015_2.ipynb
	- mnist
- cv1015_3.ipynb
	- mnist
- cv1015_4.ipynb
	- fashion_mnist
- cv1015_5.ipynb
	- jena_climate_2009_2016.csv


## 수업내용

```
통계           -->    머신러닝  -->    딥러닝 (DNN, RNN, CNN)  

데이터분석가
```

##### 딥러닝 구조
- 기둥(Dense?)을 세운다
- 학습의 시대 --> 추론의 시대
```
DNN
                기둥
입력층   -> 은닉층1(렐루)    ->  은닉층2(렐루)   ->  출력층(소프트맥스)
                  활성화함수

  --> 순전파
  <-- 역전파
```
- 은닉층을 N개 둘 수 있다
- 노드도 N개 확장 가능
- 활성화 함수 : 가중치값 보정

##### CNN
```

입력층 -> 합성곱층 -> 풀링층 -> 완전연결층 -> 출력층
  
```
- 픽셀 하나하나가 독립변수가 됨
	- 300X300 -> 90000
- 합성곱 신경망
	- 합성곱층 : 다차원배열을 1차원 배열로 flatten
- 합성곱 신경망 구조
	- 입력층
	- 합성곱층   <- DNN에서 추가된 개념(?)       특성 추출
	- 풀링층      <- DNN에서 추가된 개념(?)       크기 줄이고
	- 완전연결층                                           카테고리(?)
	- 출력층         
- 입력층 이미지는 height,width,channel 3차원 데이터  -> (300, 300, 3)
- 합성곱층과 풀링층을 거치면서 입력 이미지의 특성을 추출
- 합성곱층
	- 특성을 추출. 
	- 커널이나 필터를 사용. 필터 결과물이 특성맵. 
	- 커널은 3x3, 5x5 크기가 일반적. 
	- 스트라이드를 이용하여 이동 간격 지정
	- 채널수만큼 커널이 있음 채널이 3개면 커널도 3개. 각 채널과 연결된 커널(필터)를 합성곱하고 그 결과 3개를 합하면 특성맵 하나가 나옴
- 풀링층
	- 특성맵의 차원을 다운 샘플링하여 연산량을 감소 -> 주요 특성 벡터를 추출
	- 풀링연산:  
		- **최대풀링**을 많이 사용
		- 평균풀링 : 특성이 희미해짐
- 완전연결층을 거치면서 1차원 벡터로 변환
- 출력층
	- 소프트맥스 활성화 함수를 거쳐 최종 결과 출력. 
	- 결과의 전체합은 1이 됨
- 입력데이터
- 하이퍼파라미터
	- 필터크기, 필터 수
	- 스프라이드

##### 전이학습
- 이미지넷 같은 데이터셋으로 훈련된 모델(사전훈련된 모델(네트워크))이 가중치를 가져와 우리의 과제에 맞게 보정해서 사용


## 실습

##### cv1015_1_DNN_mnist.ipynb 

DNN을 이용하여 mnist 데이터셋으로 연습

```python
# DNN
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(10, activation="softmax")) 
```

```python
# 은닉층
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))     # 은닉층 추가 
model.add(Dense(10, activation="softmax"))  
```

```python
# dense->dropout
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(128, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(10, activation="softmax"))   # 숫자 10가지 분류
```

![[Pasted image 20261015114033.png]]


```python
# 이미지 출력
plt.imshow(X_train[0], cmap="Greys")
plt.show()
```
cmap 파라미터
- ColorMap
- 이미지 데이터를 변경하는 인수가 아니라, 데이터를 화면에 표현하는 색상 규칙
- 주요 cmap 정리

| cmap        | 특징                 | 활용 예                  |
| ----------- | ------------------ | --------------------- |
| `'gray'`    | 흑백 명암              | MNIST, OpenCV 그레이 이미지 |
| `'Greys'`   | gray와 밝기 표현 방향이 반대 | 0이 흰색, 255 검정색        |
| `'viridis'` | 보라·초록·노랑           | 데이터 분포, 특징값           |
| `'jet'`     | 파랑·초록·빨강           | 히트맵, 기존 시각화 코드        |
| `'hot'`     | 검정·빨강·노랑·흰색        | 온도, 활성화 강도            |
| `'cool'`    | 청록·자홍              | 값의 변화 시각화             |
| `'inferno'` | 검정·보라·주황·노랑        | 열 분포, 강도 시각화          |
| `'Reds'`    | 흰색에서 빨간색 계열        | 특정 영역의 강도             |
| `'Blues'`   | 흰색에서 파란색 계열        | 특정 영역의 강도             |


#### cv1015_4_fashion_mnist.ipynb

fashion-MNIST 데이터셋 이용

모델1: 입력을 바로 10개 클래스로 분류
```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(10, activation='softmax'))

# epoch = 5
# ---------------------
# 마지막 epoch
# accuracy: 0.8579 - loss: 0.4131 - val_accuracy: 0.8446 - val_loss: 0.4478
# evaluation
# accuracy: 0.8446 - loss: 0.4478

# epoch = 10
# ---------------------
# accuracy: 0.8676 - loss: 0.3816 - val_accuracy: 0.8463 - val_loss: 0.4428
# accuracy: 0.8463 - loss: 0.4428
```

모델2: 은닉층 추가
128개의 뉴런으로 이미지의 특징을 학습한 뒤 분류
```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 마지막 epoch
# accuracy: 0.8922 - loss: 0.2927 - val_accuracy: 0.8648 - val_loss: 0.3767
# evaluation
# accuracy: 0.8648 - loss: 0.3767
```

모델2: 은닉층 여러 개 추가
```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 마지막 epoch
# accuracy: 0.8762 - loss: 0.3406 - val_accuracy: 0.8678 - val_loss: 0.3628
# evaluation
# accuracy: 0.8699 - loss: 0.3574
```

모델 3. 여러 은닉층 + Dropout
많은 뉴런과 은닉층으로 복잡한 특징을 학습하고, Dropout으로 과적합을 줄이도록 설계
```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# epoch = 5
# ---------------------
# 마지막 epoch
# accuracy: 0.8762 - loss: 0.3406 - val_accuracy: 0.8678 - val_loss: 0.3628
# evaluation
# accuracy: 0.8840 - loss: 0.3291

# epoch = 10
# ---------------------
# accuracy: 0.8898 - loss: 0.2995 - val_accuracy: 0.8743 - val_loss: 0.3531
# accuracy: 0.8463 - loss: 0.4428
```

> 은닉층이 추가될수록 학습정확도는 올라가지만 테스트 정확도는 변화없음-> 과적합이 발생
> 학습 정확도는 높은데 검증 정확도가 낮다면 과적합을 의심
> 은닉층이 무조건 추가된다고 좋은 것은 아님


#### cv1015_5_CNN.ipynb

```python
model = Sequential()
model.add(Conv2D(filters=32,             # 필터 수
		          kernel_size=(3, 3),    # 필터 크기 3X3
		          ctivation='relu',      # 함성곱 결과에 RELU 적용
		          input_shape=(28, 28, 1))) # 높이28,너비28,채널1개인 흑백이미지
model.add(MaxPooling2D(pool_size=(2, 2)))  # 특성맵의 차원을 다운 샘플링하여 연산량을 감소
model.add(Dropout(0.25))    # 20~30% 사이
model.add(Flatten())        # 완전연결층
model.add(Dense(units=64, activation='relu'))   # 노드 수는 2^5(32) ~ 2^7(128())
model.add(Dropout(0.25))      #과적합 방지
model.add(Dense(units=10, activation='softmax'))  # 출력층

# epoch = 10
# ---------------------
# 마지막 epoch
# accuracy: 0.9129 - loss: 0.2384 - val_accuracy: 0.9093 - val_loss: 0.2536
# evaluation
# accuracy: 0.9093 - loss: 0.2536


# epoch = 20
# ---------------------
# 마지막 epoch
# accuracy: 0.9619 - loss: 0.1008 - val_accuracy: 0.9098 - val_loss: 0.3495
# evaluation
# accuracy: 0.9072 - loss: 0.3708


# EarlyStopping 지정  -> epoch 7에서 멈춤
# ---------------------
# accuracy: 0.9625 - loss: 0.0960 - val_accuracy: 0.9129 - val_loss: 0.3570
# accuracy: 0.9094 - loss: 0.3452

```

> DNN 보다 CNN의 정확도가 올라감  (86%  --> 90% )

###### 레이어별 구조 정리

| 코드                    | 레이어 종류   | 역할               |
| --------------------- | -------- | ---------------- |
| `Conv2D(32, (3,3))`   | 합성곱층     | 이미지의 특징 추출       |
| `MaxPooling2D((2,2))` | 풀링층      | 특징맵의 가로·세로 크기 축소 |
| `Dropout(0.25)`       | 정규화 기법   | 과적합 완화           |
| `Flatten()`           | 평탄화층     | 특징맵을 1차원 벡터로 변환  |
| `Dense(64, relu)`     | 완전연결 은닉층 | 추출된 특징을 조합하여 학습  |
| `Dropout(0.25)`       | 정규화 기법   | 과적합 완화           |
| `Dense(10, softmax)`  | 완전연결 출력층 | 10개 숫자 클래스 분류    |
```
입력이미지 흑백이미지
  ↓                 (28 X 28 X 1)
  
--------------------- 특징추출 ----------------------------------------

Conv2D — 합성곱층
            3 × 3 필터 32개
  ↓         이미지의 작은 영역을 이동하면서 선,모서리,곡선등의 특징을 학습
            특징 추출: 26 × 26 × 32
  
MaxPooling2D — 풀링층
            2 × 2 영역에서 최댓값 선택
  ↓         차원 축소: 13 × 13 × 32 
  
Dropout — 정규화 기법
            0.25
  ↓         학습 중 일부 출력을 무작위로 비활성화.
            일부 뉴런에 지나치게 의존하는 현상을 줄여 과적합을 완화.
            결과적으로 학습 데이터에만 지나치게 맞춰지는 현상을 줄이고 
            새로운 이미지에서도 잘 분류하도록 함.
            평가나 예측 단계에서는 비활성화.
                    
Flatten — 평탄화층  
            13 × 13 × 32 → 5,408개
  ↓         3차원 특징맵을 1차원 벡터로 변환. 가중치는 가지지 않음
  
--------------------- 분  류 -----------------------------------------

Dense(64) — 완전연결 은닉층
            64개 뉴런 + ReLU
  ↓         Flatten으로 변환한 5,408개의 값을 입력받아 64개의 뉴런에서 특징을 조합
  
Dropout — 정규화 기법
            0.25
  ↓ 
Dense(10) — 출력층  
            Softmax
            숫자 0~9의 클래스별 확률 출력                     
   
```
> **합성곱층이 이미지의 특징을 추출하고, 풀링층이 특징맵 크기를 줄이며, Flatten이 데이터를 펼치고, 완전연결층과 출력층이 최종 숫자를 분류합니다.**

![[CNN01.png]]

antigravity에서 colab 실행, 노트북의 GPU 사용하려면? CUDA