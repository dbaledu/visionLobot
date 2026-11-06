


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 


## 수업내용

- SimpleRNN - sin 난수 : `py0918_1_simpleRNN.ipynb`
- transformer - sin 난수: `py0918_2_트랜스포머.ipynb`
- TransformerBlock - sin 난수 : `py0918_3_트랜스포머_성능향상.ipynb`
- SimpleRNN- 주식 : `py0918_4_트랜스포머_주식.ipynb`
- SimpleRNN- 주식(변동가) : `py0918_5_주식_추세특성.ipynb`
- SimpleRNN-날씨 : `py0918_6_RNN_날씨.ipynb`

#### 용어 정리

|이름|분류||
|---|---|---|
|CNN|모델(아키텍처)||
|RNN|모델(아키텍처)||
|Transformer|모델(아키텍처)||
|SimpleRNN|레이어||
|LSTM|레이어||
|GRU|레이어||
|MultiHeadAttention|레이어|Transformer를 구성하는 핵심 레이어|
|Dense|레이어||
|Conv1D|레이어||
|LayerNormalization|레이어||
|Embedding|레이어||

## SimpleRNN

- 실습파일: py0918_1_simpleRNN.ipynb

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

#---------------------------------
# 데이터 생성 : 시계열 데이터를 만드는 전처리
#---------------------------------
# 0부터 999까지의 사인파 시계열 데이터
data = np.sin(np.arange(1000)* 0.02)  # 0.02가 클수록 파형이 더 촘촘해지고, 작을수록 파형이 길어집니다.

x = [] # 시퀀스(=독립변수, feature)
y = [] # 타겟

for i in range(len(data)-20):  # 시퀀스길이 20으로 windows 나누기
  x.append(data[i:i+20])   # 인덱스 하나씩 이동하면서 20개씩 자르기
  y.append(data[i+20])     # 21번째가 타겟임.  assuming this was the intent to populate 'y'

#---------------------------------
# 데이터로 구조 확인하고 RNN 입력데이터 형식(차원)으로 변환
#---------------------------------
X = np.array(x) # Convert list of arrays to a single numpy array
y = np.array(y) # Convert list to a numpy array for consistency

# x 변수의 타입과 dimention, shape 출력
print(f"x 변수의 타입: {X.dtype}")
print(f"x 변수의 차원 (ndim): {X.ndim}")
print(f"x 변수의 shape: {X.shape}")

print(f"\ny 변수의 타입: {y.dtype}")
print(f"y 변수의 차원 (ndim): {y.ndim}")
print(f"y 변수의 shape: {y.shape}")

# simpleRNN : RNN(LSTM, GRU)에서는 입력 데이터가 반드시 3차원 형태 (샘플 수, 시퀀스 길이, 특성 수)  <-- 시퀀스길이 가 타임스탭 ?
X = X[..., np.newaxis]
print("차원 변경후 x 변수의 shape: ", X.shape)

#---------------------------------
# 학습데이터, 테스트 데이터 분리
#---------------------------------
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

#---------------------------------
# 모델 생성. SimpleRNN 레이어 사용
#---------------------------------
model = Sequential([
    SimpleRNN(32, input_shape=(20,1)),  # 기본으로 활성화함수는 tanh(하이퍼볼릭 탄젠트) 함수가 적용됨
    Dense(1, activation="linear")       # linear 는 회귀. 생략해도됨.
])

model.compile(optimizer='adam', loss='mse')  # 회귀에는 adam을 주로 사용
model.summary()

#---------------------------------
# 모델 학습
#---------------------------------
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#---------------------------------
# 예측 결과 시각화
#---------------------------------
pred = model.predict(X_test)

plt.plot(y_test, label='actural')
plt.plot(pred , label='Prediction')
plt.legend()
plt.show()

#-----------------------------------------
# 예측 : 이전 데이터 20개로 다음 값을 예측
#-----------------------------------------
recent_data = data[-20:]  # 이전 데이터 20개
recent_data_3d = recent_data.reshape(1,20,1)
pred = model.predict(recent_data_3d)
print(pred)
print('1000번째 예측값', pred[0][0])
print(data[-1])
```

### 데이터 생성 : 시계열 데이터를 만드는 전처리

```python
# 0부터 999까지의 사인파 시계열 데이터
# 0.02가 클수록 파형이 더 촘촘해지고, 
#        작을수록 파형이 길어집니다.
data = np.sin(np.arange(1000)* 0.02)  

x = [] # 시퀀스(=독립변수, feature)
y = [] # 타겟

for i in range(len(data)-20):  # 시퀀스길이 20으로 windows 나누기
  x.append(data[i:i+20])   # 인덱스 하나씩 이동하면서 20개씩 자르기
  y.append(data[i+20])     # 21번째가 타겟임.  assuming this was the intent to populate 'y'
```

RNN(LSTM, GRU, SimpleRNN)이 학습할 수 있는 형태로 **시계열 데이터**를 만드는 전처리 과정입니다. 연속된 데이터를 일정길이의 시퀀스(sequence) 로 나누어 학습합니다.

> 사인파 시계열을 20개씩 묶어서 입력(X)으로 만들고, 그 다음(21번째) 값을 정답(y)으로 만드는 **Sliding Window(이동 윈도우)** 작업

![[0928_3SimpleRNN_시계열.png]]

![[0928_1.png]]



- 980개의 학습 데이터
- 각 데이터마다 20개의 시간 정보

**라디안과 디그리(각도) 변환**

- 1 rad = 180 / [**π](https://ko.wikipedia.org/wiki/%CE%A0) ←** 180도를 파이로 나눈 값 = 57.3도
- 사인 함수는 2[**π](https://ko.wikipedia.org/wiki/%CE%A0)** (=6.283) 라디안마다 한 주기를 반복

### RNN 입력데이터 형식(차원)으로 변환

```python
X = np.array(x)         # list를 ndarray로 변환
X = X[..., np.newaxis]  # 차원 추
```

RNN(LSTM, GRU)에서는 입력 데이터가 반드시 **3차원 형태 (샘플 수, 시퀀스 길이, 특성 수)**

### 학습데이터, 테스트 데이터 분리

```python
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]
```

train_test_split을 이용할 때는 "shuffle=False" 지정  
<- 시계열 데이터이므로 무작위로 섞이면 안됨.

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, **shuffle=False**)

### SimpleRNN 레이어 사용

```python
model = Sequential([
    SimpleRNN(32, input_shape=(20,1)),  # 기본으로 활성화함수는 tanh(하이퍼볼릭 탄젠트) 함수가 적용됨
    Dense(1, activation="linear")       # linear 는 회귀. 생략해도됨.
])

model.compile(optimizer='adam', loss='mse')  # 회귀에는 adam을 주로 사용
model.summary()
```

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Layer (type)           ┃ Output Shape  ┃ Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ simple_rnn (SimpleRNN) │ (None, 32)    │   1,088 │
├────────────────────────┼───────────────┼─────────┤
│ dense (Dense)          │ (None, 1)     │      33 │
└────────────────────────┴───────────────┴─────────┘
```

**SimpleRNN 파라미터**

- **입력가중치**: input feature * Units -> 1 * 32 = **32**
- **순환가중치**: units * units -> 32 * 32 = **1024**
- **바이어스**: units -> **32**

파라미터 : 32 + 1024 + 32 = **1088**

![[0928_2.png]]

**순환가중치**

이전 프레임의 32가지 기억이 현재의 32가지 기억을 어떻게 업데이트할지 결정하는 규칙. 은닉층에 뉴런이 32개가 있는 경우 이전의 32개 뉴런이 현재의 32개 뉴런과 모두 연결되어서 파라미터는 `32×32` 가 필요

### LSTM 레이어 사용

```python
model = Sequential([  
    LSTM(32, input_shape=(20,1)),  # 기본으로 활성화함수는 tanh(하이퍼볼릭 탄젠트) 함수가 적용됨
    Dense(1, activation="linear")       # linear 는 회귀. 생략해도됨.
])
```

#### GRU 레이어 이용

```python
model = Sequential([
    GRU(32, input_shape=(20,1)),
    Dense(1, activation="linear")       # linear 는 회귀. 생략해도됨.
])
```

### RNN 계열 레이어 비교

|레이어|기억 방식|장점|단점|사용 예|
|---|---|---|---|---|
|`SimpleRNN`|이전 은닉 상태만 기억|구조가 단순, 학습용으로 좋음|긴 문맥 기억 어려움|사인파 예측|
|`LSTM`|Cell State + Gate|장기 기억에 강함|파라미터 많음|번역, 시계열,|
|긴 주행 기록 분석|||||
|`GRU`|Update/Reset Gate|LSTM보다 빠름|아주 긴 문맥은 LSTM보다 약할 수 있음|음성, 센서 시계열(IMU, 속도)|
|`Bidirectional`|정방향+역방향|앞뒤 문맥 모두 활용|실시간 처리 어려움|문장 분류, 문장·음성 이해|
|`CNN + LSTM`||||카메라 영상의 프레임 연속 분석|

특히 자율주행에서는 카메라 영상 → CNN으로 특징 추출 → LSTM/GRU로 시간 흐름 학습하는 구조가 많이 사용됩니다. 예를 들어 `Conv1D → MaxPooling1D → LSTM` 모델은 연속된 센서 데이터나 영상 특징의 변화를 기억하며 조향이나 속도 예측에 활용되는 대표적인 구조입니다.

---

## transformer 모델

- 실습파일 : `py0918_2_트랜스포머.ipynb`

앞에서 했던 sin 시계열 데이터 이용함.

```python
#---------------------------------
# 데이터 생성
#---------------------------------
# 0부터 999까지의 사인파 시계열 데이터
data = np.sin(np.arange(1000)* 0.02)  # 0.02가 클수록 파형이 더 촘촘해지고, 작을수록 파형이 길어집니다.

x = [] # 시퀀스(=독립변수, feature)
y = [] # 타겟

for i in range(len(data)-20):  # 시퀀스길이 20으로 windows 나누기
  x.append(data[i:i+20])   # 인덱스 하나씩 이동하면서 20개씩 자르기
  y.append(data[i+20])     # 21번째가 타겟임.  assuming this was the intent to populate 'y'

X = np.array(x) # Convert list of arrays to a single numpy array
y = np.array(y) # Convert list to a numpy array for consistency

X = X[..., np.newaxis]

#---------------------------------
# 학습데이터, 테스트 데이터 분리
#---------------------------------
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

#---------------------------------
# 트랜스포머 모델 생성. 레이어 지정
#---------------------------------
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

inputs = Input(shape=(20,1))

# 1. 입력을 Transformer가 사용할 임베딩 차원으로 변환
x = Dense(32)(inputs)

# 멀티헤드 어텐션 층
attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)

# 정규화(어텐션을 추출하여 입력값과 연산(Residual Connection) -> 특징 부여)
x = LayerNormalization()(x + attention_output)

# 최대 풀링 층. 출력층을 위한 평탄화(시간 차원 축소)
x = GlobalAveragePooling1D()(x)

# 회귀 출력층
outputs = Dense(1)(x)

# 모델 생성 (output만 넣어주면 거꾸로 추적해서 모델을 완성. x는 중간노드라서 전달할 필요 없음)
# output <- Dense <- GlobalAveragePooling1D <- LayerNormalization <- MultiHeadAttention <- Dense
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse')  # 회귀에는 adam을 주로 사용
model.summary()

#---------------------------------
# 모델 학습
#---------------------------------
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

#---------------------------------
# 예측 결과 시각화
#---------------------------------
pred = model.predict(X_test)

plt.plot(y_test, label='actural')
plt.plot(pred , label='Prediction')
plt.legend()
plt.show()
```

### 트랜스포머 모델 생성. 레이어 지정

```python
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

inputs = Input(shape=(20,1))

# 1. 입력을 Transformer가 사용할 임베딩 차원으로 변환
x = Dense(32)(inputs)

# 멀티헤드 어텐션 층
attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)

# 정규화(어텐션을 추출하여 입력값과 연산(Residual Connection) -> 특징 부여)
x = LayerNormalization()(x + attention_output)

# 최대 풀링 층. 출력층을 위한 평탄화(시간 차원 축소)
x = GlobalAveragePooling1D()(x)

# 회귀 출력층
outputs = Dense(1)(x)

# 모델 생성 (output만 넣어주면 거꾸로 추적해서 모델을 완성. x는 중간노드라서 전달할 필요 없음)
# output <- Dense <- GlobalAveragePooling1D <- LayerNormalization <- MultiHeadAttention <- Dense
model = Model(inputs=inputs, outputs=outputs)
```

각 레이어 역할

```python
inputs = Input(shape=(20,1))
```

```python
# 멀티헤드 어텐션 층
attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
```

```python
# 정규화(어텐션을 추출하여 입력값과 연산(Residual Connection) -> 특징 부여)
x = LayerNormalization()(x + attention_output)
```

```python
# 최대 풀링 층. 출력층을 위한 평탄화(시간 차원 축소)
x = GlobalAveragePooling1D()(x)
```

```python
# 회귀 출력층
outputs = Dense(1)(x)
```

```python
model = Model(inputs=inputs, outputs=outputs)
```

#### 구조 비교

|LSTM|Transformer|
|---|---|
|이전 정보를 순서대로 기억|모든 시점을 동시에 비교|
|`LSTM(32)`|`MultiHeadAttention(num_heads=4, key_dim=8)`|
|마지막 은닉 상태 사용|`GlobalAveragePooling1D`로 전체 정보 사용|
|긴 시퀀스에서 정보 손실 가능|장거리 의존성 학습에 강함|