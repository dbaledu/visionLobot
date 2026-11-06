


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 


- 가상머신
    - 회귀 Regression :
        - Linear Regression . 예측결과 숫자값. 하나 이상의 **독립변수(Feature, X)**를 이용하여 연속적인 숫자인 **종속변수(Target, y)**를 예측하는 지도학습의 **회귀 알고리즘**
        - Logistic Regression : 예/아니오 판단(이진분류)
    - 분류 Classification :

### 단순선형회귀

```python
y = ax + b

형태의 관계를 학습합니다.

여기서 a는 기울기(회귀계수), b는 절편입니다.
```

![[0904_1.png]]

```python
데이터 구조
sklearn.utils._bunch.Bunch
  - feature_names
  - data
  - target_names
  - target
  
```

산점도

행열이 같은 항목인 경우는 히스토그램으로 표현. 정규분포를 확인할 수 있음

![[0904_2.png]]

## Logistic Regression

Entropy - 혼잡도?

One-Hot Encoding

- 클래스 종류가 독립변수가 됨. 클래스가 늘어나면 열의 수가 늘어남

![[0904_3.png|250]]

### py0904_linearLogistic.ipynb

- fetch_california_housing() : 캘리포니아 주택가격
- LinearRegression

#### py0904_linearRegression.ipynb : 캘리포니아 주택가격 맞추기

전체소스

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns           # matplotlib 확장

#==================================
# 선형회귀 : 캘리포니아 주택가격
# 주택가격 -> 종속변수
# 주택가격에 영향을 많이 미치는 정보 - 하이퍼파라미터,   영향을 미치는 정보 - 파라미터
#==================================
cali = fetch_california_housing()   # 캘리포니아 주택가격
print(type(cali))                # <class 'sklearn.utils._bunch.Bunch'>
#print(cali)
print(cali.feature_names)       # 데이터 컬럼리스트
print(cali.target_names)        # 타겟 컬럼리스트
#print(cali.data)               # 데이터(독립변수)
df = pd.DataFrame(cali.data, columns=cali.feature_names)
df['target'] = cali.target  # 주택가격(종속변수)

# data scaling이 되어 있음 (크기와 단위를 맞춰야 함)
df.head()
#df.info()

# 1000개만 샘플링해서 상관관계 확인
sns.pairplot(df.sample(1000))
plt.title('data의 다차원 plot')
plt.show()
# x축과 y축이 같은 항목이면 히스토그램으로 표현 -> 정규분포 확인
# 이상치 확인

df.plot(kind='scatter', x='Longitude', y='Latitude', alpha=1.0, figsize=(12,10))  # figsize 는 인치 단위
df.show()

# 독립변수
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
X =df[ ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',  'Latitude', 'Longitude'] ]
# X = df[[
#     'MedInc'
# ]]
#X = df.drop('target', axis=1)

# 종속변수
y = df['target']

# 훈련용 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.8,
    test_size=0.2,   # train_size만 지정하면 됨. 생략가능
    random_state=42
)

# 선형회귀모델 생성
model = LinearRegression()

# 선형회귀모델 학습
model.fit(X_train, y_train)

# 훈련 성능 평가
train_score = model.score(X_train, y_train)
print('훈련 데이터 점수: ', train_score)

# 훈련 데이터와 테스트 데이터가 잘 분리되었는지 검증
# 이렇게 여러 번 평가하면 특정 데이터 분할에서 우연히 잘 나온 것인지, 대체로 안정적으로 성능이 나오는 모델인지 확인할
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,   # 평가할 머신러닝 모델
                         X_train, # 독립변수(훈련데이터로 검)
                         y_train, # 종속변수
                         cv=10,   # 데이터를 10개 Fold로 나눔. 10변 교차검
                         scoring='neg_mean_squared_error')     # MSE를 평가 기준으로 사용. scoring은 함수 이름
print(f'MSE 점수:{scores}')
print(f'NMSE 점수 평균:{scores.mean()}')
print(f'NMSE 점수 표준편차:{scores.std()}')

# 교차검증(corss validation) - 회귀모델 신뢰성 평가
r2_score = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
print(f'R2 점수 평균:{r2_score.mean()}')

from sklearn.metrics import mean_squared_error
# Test 데이터를 별도로 남겨두었다가 마지막에 객관적인 성능을 확인

# 테스트 데이터 예측
y_pred = model.predict(X_test)
# 손실함수
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#검증
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R2:', r2)

# 테스트 데이터를 사용한 시각화
predicted = model.predict(X_test)   # 훈련시키지 않은 테스트 데이터
expected = y_test   # 훈련안된 정답
plt.figure(figsize=(8,4))
plt.scatter(expected, predicted)  # x:실제가격 y:예측가격
plt.plot([0, 5], [0, 5], '--r')   # 빨간선 그래프 - 이상적인 값
plt.xlabel('True proce(($100,000s))')
plt.ylabel('predicted price($100,000s)')
plt.tight_layout()
plt.show()

```

### py0904_2_logistic.ipynb

- LogisticRegression
- admission.csv : 지원자의 시험 성적 등을 이용해 합격 여부를 예측

전체소스

```python
#============================
# 로지스틱회귀 : 2개 중에서 선택(T/F)  - 시그모이드 활성화 함수
#============================
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 지원자의 시험 성적 등을 이용해 합격 여부를 예측하는 데이터
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/admission.csv')
df

# 판다스를 사용한 라벨 인코딩
import numpy as np
from sklearn.preprocessing import LabelEncoder

items1 = pd.DataFrame(['TV','TV','냉장고','컴퓨터','냉장고','컴퓨터','에어콘'])
print(items1)

items1 = items1[0].map({'TV':0, '냉장고':1, '에어콘':2, '컴퓨터':3})
print("============")
print(items1)

# 사이킷런을 사용한 라벨 인코딩
import numpy as np
from sklearn.preprocessing import LabelEncoder

items2 = ['TV','TV','냉장고','컴퓨터','냉장고','컴퓨터','에어콘']
le = LabelEncoder()
items2Encod = le.fit_transform(items2)
print(items2Encod)
print(le.classes_)  # 사이킷런에서 인식한 클래스명 순서

# 사이킷런을 사용한 원-핫 인코딩
from sklearn.preprocessing import OneHotEncoder

items3 = pd.DataFrame(['TV','TV','냉장고','컴퓨터','냉장고','컴퓨터','에어콘'])
one = OneHotEncoder(sparse_output=False)
items3Encod = one.fit_transform(items3)
print(items3)
print(" TV 냉장고 에어콘 컴퓨터 ")
print(items3Encod)

X = df.drop('admit', axis = 1)
y = df['admit']
display(X.head())
display(y.head())

X = df[['gre','gpa','rank']].values  # 독립변수
y= df['admit'].values                # 종속변수
print(X)
print('=========')
print(y)

# 과적합을 막기 위해 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 로지스틱 회귀 모델 생성
model = LogisticRegression()

# 생성된 모델로 데이터 훈련
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# gre, gpa, rank
pred_val = [[500, 4.1, 1]]
res = model.predict(pred_val)
print('결과 예측:', res[0])

myScore_str = input('본인의 GRE, GPA, RANK를 입력하세요 (예: 700 4.1 2)')

try:
  score = myScore_str.split()
  gre = int(score[0])
  gpa = float(score[1])
  rank = int(score[2])
  myscore_pred = [[gre, gpa, rank]]
  res = model.predict(myscore_pred)
  if res == 1:
    print("당신은 하버드 대학에 합격하였습니다.")
  else:
    print("당신은 하버드 대학에 불합격하였습니다.")

except(ValueError, IndexError) :
  print("잘못된 형식 입력입니다. GRE, GPA, RANK를 공백으로 구분하여 입력해주세요 (예: 700 4.1 2)")
```

### 0904_3_logistic_iris.ipynb

평가

|평가 지표|의미|일반적인 해석|
|---|---|---|
|**MAE**|실제값과 예측값 차이의 절댓값 평균|작을수록 좋음|
|**MSE**|오차를 제곱한 값의 평균|작을수록 좋음|
|**R²**|모델이 종속변수의 변동을 설명하는 정도|일반적으로 1에 가까울수록 좋음|

전체소스

```python
#============================
# 로지스틱회귀 : 여러개 중에서 선택
#============================
from sklearn.datasets import load_iris
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

iris = load_iris()     # bunch type
iris.feature_names
#iris.target_names   # ['setosa', 'versicolor', 'virginica'],
# iris.data
# iris.target
# len(iris.data)

# np.c_ : 2개의 배열을 세로로 연결
iris_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns = iris['feature_names'] + ['target'] )
iris_data['target'] = iris_data['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

iris_data

# from re import X
# 데이터 분리
X_data = iris_data.iloc[:, :-1]
Y_data = iris_data.iloc[:,[ -1]]   # loc[:, ['target']]

type(Y_data)
print(X_data)

#===================================================
# sepal length (cm)'  'sepal width (cm)
#===================================================
# 2개의 독립변수만 사용  --> 차트로 그리기 위해
X = X_data.iloc[:, [0,1]].values    #  'sepal length (cm)'  'sepal width (cm)'  2개의 컬럼만 추출

#label encoding
y = Y_data['target'].map({'setosa':0, 'versicolor':1, 'virginica':2})
print(X.ndim, X.shape)
print(y.ndim, y.shape)   # 영상에서는 차원 변화 필요함 1차원으로 변경

# shape 속성 결과
# (m, ) : 길이가 m인 1차원 (열백터)
# (행,열) : 행X열 2차원 (matrix)
# (면,행,열) : 높이X행X열 3차원 (tensor)  # 칼라이미지 데이터는 3차원

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
model = LogisticRegression()

parameters = {
    'C': [ 2 ** 0, 2 ** 3, 2 ** 6, 2 ** 9, 2 ** 12 ],
    'random_state' : [0, 7, 12, 42]
}

# 교차 검증과 하이퍼파라미터 최적화 동시에 처리   파라미터를 변경하면서 최적점수 확인
gs = model_selection.GridSearchCV(model, parameters)
gs.fit(X, y)
print(gs.best_estimator_)
print(f'최적 점수:{gs.best_score_}')
print(f'최적 파라미터:{gs.best_params_}')
model = gs.best_estimator_  # 최적화된 모델

model.fit(X, y)
h = .02
x_min, x_max = X[:,0].min() - .5 , X[:,0].max() + .5   # x축 범위
y_min, y_max = X[:,1].min() - .5 , X[:,1].max() + .5   # y축 범위
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Set3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.Set3)
plt.xlabel=('Sepa; length')
plt.ylabel=('Sepa; width')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())  # Disable xticks.
# plt.yticks(())
plt.show()

#===================================================
# petal length (cm)', 'petal width (cm)
#===================================================
# 2개의 독립변수만 사용  --> 차트로 그리기 위해
X = X_data.iloc[:, [1,3]].values    #  'sepal length (cm)'  'sepal width (cm)'  2개의 컬럼만 추출

#label encoding
y = Y_data['target'].map({'setosa':0, 'versicolor':1, 'virginica':2})

# 로지스틱 회귀 모델 생성
model = LogisticRegression()

# 교차 검증과 하이퍼파라미터 최적화
parameters = {
    'C': [ 2 ** 0, 2 ** 3, 2 ** 6, 2 ** 9, 2 ** 12 ],
    'random_state' : [0, 7, 12, 42]
}
gs = model_selection.GridSearchCV(model, parameters)
gs.fit(X, y)

print(gs.best_estimator_)
print(f'최적 점수:{gs.best_score_}')
print(f'최적 파라미터:{gs.best_params_}')
model = gs.best_estimator_  # 최적화된 모델

```