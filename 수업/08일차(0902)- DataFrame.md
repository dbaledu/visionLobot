


## 수업목표
- DataFrame
- 시각화(matplotlib, seaborn)
- 데이터분석

## 실습파일
- ㅇㅇ
- 


1. dictionary 로 DataFrame 생성

```python
import pandas as pd  # 세션이 다시 시작되면 임포트해야 함
# 딕셔너리
data = {
     '2015':[9904312, 3448737, 2890451, 2466502],
     '2010':[9631482, 3393191, 2632035, 2431774],
     '2005':[9762546, 3512547, 2517680, 2465601],
     '2000':[9853972, 3655437, 2466338, 2473990],
     '지역' : ['수도권','경상권','전라권','강원권'],
     '2010-2015증가율': [0.0283, 0.0163, 0.0982, 0.0141]    
         }       
df = pd.DataFrame(data) 
```

1. 행 인덱스 부여

```python
# 행 인덱스 부여. 생략하면 0부터 순차적으로 부여 됨        
index = ['서울','부산','인천','대구']         
df = pd.DataFrame(data, index)  
```

1. 리스트를 이용하여 생성

```python
# 컬럼명을 따로 지정해야함
data = [
    [9904312, 9631482, 9762546, 9853972, '수도권', 0.0283],
    [3448737, 3393191, 3512547, 3655437, '경상권', 0.0163],
    [2890451, 2632035, 2517680, 2466338, '전라권', 0.0982],
    [2466502, 2431774, 2465601, 2473990, '강원권', 0.0141]
]
df = pd.DataFrame(data, columns=['2015', '2010', '2005', '2000', '지역', '2010-2015증가율'])
```

1. 컬럼값 변경. 브로드캐스트 연산 이용

```python
df['2010-2015증가율'] = df['2010-2015증가율'] * 100
```

1. 새로운 컬럼 추가

```python
df['2010-2015 인구증가율'] =( (df['2010'] - df['2005'])/df['2005'] * 100 ).round(2)

```

1. 하나의 컬럼 인덱싱 → Series 변환

딥러닝에서 정답은 series 1차원. 학습 데이터는 2차원. 데이터에서 정답만 추출할 때 이용

```python
# 하나의 컬럼 indexing -> Series 반환
print(df['지역'])
print(type(df['2015']))
```

1. 여러 개의 컬럼 슬라이싱

```python
# 여러 개의 컬럼  slicing  -> 부분 dataFrame 반환
print(df[['2010년', '2015년']])
print(type(df[['2010년', '2015년']]))
```

- `[]`는 컬럼 선택용인데, 슬라이스가 들어오면 예외적으로 행 선택으로 동작
- 애매함 때문에 명확하게 `loc`/`iloc`을 쓰라고 권장

|표현|안에 들어간 것|결과|
|---|---|---|
|`df['지역']`|단일 문자열|지역 컬럼 (Series)|
|`df[['2010', '2015']]`|리스트|해당 컬럼들 (DataFrame)|
|`df['서울':'부산']`|슬라이스(`:`)|서울~부산 행들|

1. loc, iloc

- loc : label
- iloc : index . Can only index by location with a integer

```python
df.loc[:, ['2010년', '2015년']]   # 열 선택 명확
df.loc['서울':'부산']              # 행 선택 명확
```

```python
# loc
print(type( df.loc['서울']))   # <class 'pandas.core.series.Series'>
print('loc["서울"]', df.loc['서울'])
print()
# iloc
print('iloc[0]', df.iloc[0])
print()

# values
print(type(df.iloc[0].values))   # <class 'numpy.ndarray'>
print('iloc[0].values', df.iloc[0].values)
```

**`df.iloc[0, 1] df.iloc[0][1] 차이점`**

```python
print('iloc[0][1]', df.iloc[0][1])
print('iloc[0,1]', df.iloc[0,1])
```

결과값은 같지만, 동작 방식은 완전히 달라요.

**`df.iloc[0, 1]`** — 한 번에 처리되는 인덱싱

- `(행, 열)` 튜플을 한 번에 받아서 바로 해당 위치의 값을 찾아요.
- 내부적으로 한 번의 연산만 일어나요. **권장되는 방식**입니다.

**`df.iloc[0][1]`** — 체이닝(연쇄) 인덱싱

- 먼저 `df.iloc[0]`이 실행돼서 **0번 행 전체가 하나의 Series로 반환**돼요.
- 그다음 그 Series에 `[1]`을 적용해서 1번째 위치의 값을 꺼내는 거예요.
- 즉 **두 번의 별도 연산**이 순차적으로 일어나는 겁니다.

```python
df.iloc[0][1] = 100   # 경고 발생 가능 (SettingWithCopyWarning)
df.iloc[0, 1] = 100   # 안전하게 원본에 값 수정됨
```

체이닝 방식으로 값을 **수정**하려고 하면, 중간에 생성된 Series가 원본의 복사본인지 뷰(view)인지 애매해져서 pandas가 경고를 띄우거나 실제로 원본이 수정되지 않을 수 있어요.

SettingWithCopyWarning:  
A value is trying to be set on a copy of a slice from a DataFrame

**결론**: 값을 읽을 때든 수정할 때든 `df.iloc[0, 1]`처럼 **콤마로 한 번에 인덱싱**하는 걸 습관화하는 게 안전해요.

[NumPy: the absolute basics for beginners — NumPy v2.5 Manual](https://numpy.org/doc/stable/user/absolute_beginners.html#indexing-and-slicing)

```python
# filter specific rows
above_35 = titanic[titanic["Age"] > 35]   # 조건식
class_23 = titanic[titanic["Pclass"].isin([2, 3])]
class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
age_no_na = titanic[titanic["Age"].notna()]   # 값이 결측값(NA: Not Available 또는 Not Applicable)이 아닌지 여부

# select specific rows and columns
adult_names = titanic.loc[titanic["Age"] > 35, "Name"]  # ,를 기준으로 첫번째는 행 조건, 두번째는 열 조건
titanic.iloc[9:25, 2:5]     # 슬라이싱
```

### 응용문제 풀이

#### 응용문제1

![[0902_1.png]]

- index를 지정하지 않으면 0부터 지정되고 지정하면 행 라벨이 됨
- dictionary 키가 dataFrame의 컬럼명이 됨

```python
# 인덱싱, 슬라이싱
df['수학']             # 문자열은  하나의 컬럼 인덱싱.          결과는 Series     
df.loc[ : , '수학' ]   # 위와 동일  모든행의 수학 컬럼.         결과가 컬럼 하나면 Series   
df[['국어', '영어']]   # [ 문자열... ]  여러 개 컬럼 슬라이싱.  결과는 dataFrame
```

```python
# 하나의 컬럼 인덱싱
df['a']                # 컬럼명
df.loc[ : , 'a']       # loc 메서드
df.iloc[:, 0]          # iloc 메서드
df.c                   # . 연산
```

```python
# 새로운 컬럼 추가
df.loc[:, '학생평균'] = df.mean(axis=1)   # axis=1  열추가
df.loc['과목평균'] = df.mean(axis=0)      # axis=0  행추
```


![[0902_2.png]]
#### 응용문제2

![[0902_3.png]]

```python
# a열과 b열 슬라이싱
df[['a','b']]             # 열 인덱싱
df.loc[ :, ['a','b']]     # loc  : 행은 전체, 열은 컬럼명 리스트
df.loc[ :, 'a':'b']       # loc  : 행은 전체, 열은 컬럼명 슬라이싱
df.iloc[:, [0, 1]]        # iloc : 행은 전체, 열은 열인덱스 리스트
df.iloc[:, 0:2]           # iloc : 행은 전체, 열은 열인덱스 슬라이싱 
```

**행: 0~1행, 열: a,c열만 추출**

![[0902_4.png]]

```python
df.loc[0:1, ['a','c']]   # [주의] loc에서는 end가 포함
                         # 행인덱스명이 0,1,2,3임  
df.iloc[0:2, [0,2]]      # iloc에서는 end가 포함 안 됨
df.iloc[[0,1], [0,2]]
```

```python
# where 사용한 경우
df.iloc[np.where(df['c'] == 'a')].loc[:,'a']

# 불리언 인덱싱을 사용한 경우  -->  성능이 더 빠름
df.loc[df['c'] == 'a', 'a']) 
```

|**구분**|**df.iloc[np.where(df['c'] == 'a')].loc[:,'a']**|**df.loc[df['c'] == 'a', 'a']**|
|---|---|---|
|**방식**|NumPy의 인덱스 추출 → `iloc` → `loc` (3단계 연산)|Pandas 조건부 슬라이싱 (1단계 연산)|
|**가독성**|복잡하고 직관적이지 않음|매우 직관적이고 단순함|
|**속도/성능**|상대적으로 느림 (임시 객체 생성 및 복사 발생 가능)|C언어 기반 내부 최적화로 빠름|
|**경고 가능성**|`SettingWithCopyWarning` 발생 가능성 있음|안전함|

### 1. 첫 번째 구문의 동작 과정 (복잡한 방식)

`df.iloc[np.where(df['c'] == 'a')].loc[:,'a']`

1. **`df['c'] == 'a'`**: 불리언(True/False) 시리즈 생성
2. **`np.where(...)`**: True인 위치의 정수 인덱스(0, 1, 2...)를 튜플 형태로 추출
3. **`df.iloc[...]`**: 추출한 정수 인덱스로 행을 1차 필터링하여 **임시 데이터프레임 생성**
4. **`.loc[:, 'a']`**: 생성된 임시 데이터프레임에서 'a' 컬럼만 최종 추출

> **문제점**: 한 번에 처리할 수 있는 작업을 NumPy를 거쳐 `iloc`과 `loc`을 연속으로 호출하면서 **불필요한 메모리 할당과 데이터 복사**가 일어납니다.

### 2. 두 번째 구문의 동작 과정 (권장 방식)

`df.loc[df['c'] == 'a', 'a']`

1. **`df.loc[행 조건, 열 이름]`**: `df['c'] == 'a'` 조건을 만족하는 행과 'a' 컬럼의 교집합을 **단 한 번의 연산**으로 직접 찾아냅니다.

### 그렇다면 `np.where`는 왜 사용하는가?

질문하신 코드처럼 **단순히 조건에 맞는 데이터프레임의 행/열을 추출할 때는 `np.where`를 쓸 필요가 없습니다.**

### **np.where을 사용하는 이유**

#### ① 조건에 따라 새로운 값을 할당할 때 (if-else 대체)

판다스 데이터프레임에서 "조건이 참이면 X, 거짓이면 Y" 형태로 새로운 컬럼을 만들 때 가장 자주 쓰입니다.

Python

```
import numpy as np

# c 컬럼이 'a'이면 1, 아니면 0을 새로운 컬럼에 할당
df['result'] = np.where(df['c'] == 'a', 1, 0)
```

#### ② 정수 위치(Index) 자체가 필요할 때

데이터의 '값'이나 데이터프레임 자체가 아니라, 조건에 맞는 데이터가 "몇 번째 행(0, 1, 2...)"에 있는 지 위치 번호(Int)가 필요한 경우에 사용합니다.

Python

```
# 조건에 맞는 행의 위치(정수 인덱스) 번호 배열 반환
row_indices = np.where(df['c'] == 'a')[0]
```

#### ③ 복수의 조건으로 값을 대체/변환할 때

NumPy 배열 수준에서 빠르고 가볍게 값 변환 처리를 하고 싶을 때 활용됩니다.

### 요약

- 데이터프레임에서 **조건으로 데이터를 조회/추출**할 때는 항상 **`df.loc[조건, 컬럼]`** 형태를 사용하세요.
- `np.where`는 조회용이 아니라 "조건부 값 생성(if-else)"이나 "정수 위치 번호 추출"이 필요할 때 사용하는 것이 올바른 용법입니다.



# matplotlib

#### colab 한글폰트 설치

```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

한글 폰트 적용

```python

import matplotlib.pyplot as plt
import numpy as np
# 한글 폰트 지정
plt.rc('font', family='NanumBarunGothic')
```

[Quick start guide — Matplotlib 3.11.1 documentation](https://matplotlib.org/stable/users/explain/quick_start.html#quick-start)

구성요소

![[0902_5.png]]

- axes : 실제 그래프가 그려지는 전체 영역
- axis : Axes 안에 있는 x축 또는 y

```python
Figure    ← 전체 그림 
└── Axes  ← 실제 그래프가 그려지는 영역(그래프 하나를 관리하는 객체)
    ├── XAxis  ← x축
    └── YAxis  ← y축
```

#### 예제

```python
plt.title('Matplotlblib Display')
plt.plot([1,4,9,6])  # list 또는 ndarray
plt.show()
```

```python
plt.plot([9,16,4,1], c='k', lw=3, ls=':', marker='s', mec='m', mew = 5, mfc='c')
```

#### 옵션

|옵션|의미|값 설명|
|---|---|---|
|`c='k'`|선 색상(color)|`k` = black(검정)|
|`lw=3`|선 굵기(linewidth)|3pt|
|`ls=':'`|선 스타일(linestyle)|`:` = 점선(dotted)|
|`marker='s'`|마커 모양|`s` = square(네모)|
|`mec='m'`|마커 테두리 색(markeredgecolor)|`m` = magenta(자홍색)|
|`mew=5`|마커 테두리 굵기(markeredgewidth)|5pt|
|`mfc='c'`|마커 내부 색(markerfacecolor)|`c` = cyan(청록색)|

#### 색상코드

#### `b`=blue, `g`=green, `r`=red, `c`=cyan, `m`=magenta, `y`=yellow, `k`=black, `w`=white

#### subplot 이용

```python
fig, ax = plt.subplots()
ax.plot()
ax.scatter()
ax.bar()

ax.set_title()
ax.set_xlabel()
ax.set_ylabel()

ax.legend()
ax.grid()
```

# seaborn

[Overview of seaborn plotting functions — seaborn 0.13.2 documentation](https://seaborn.pydata.org/tutorial/function_overview.html)

차트 종류

### relplot (관계, relational)

두 변수 사이의 **관계**를 보여줄 때 사용해요.

- **scatterplot**: 두 연속형 변수의 관계를 점으로 표현. 상관관계, 군집, 이상치 파악에 유용
- **lineplot**: 시간 흐름 등 순서가 있는 데이터의 추세를 선으로 표현. 시계열 데이터, 센서 값 변화 등에 적합

### displot (분포, distributions)

한 변수(또는 두 변수)가 **어떻게 분포되어 있는지**를 보여줄 때 사용해요.

- **histplot**: 구간(bin)별로 데이터 개수를 막대로 표현하는 히스토그램. 가장 기본적인 분포 파악 도구
- **kdeplot**: 히스토그램을 부드러운 곡선으로 근사한 밀도 추정 그래프. 분포의 형태(봉우리, 치우침)를 더 매끄럽게 확인
- **ecdfplot**: 누적 분포 함수(경험적 CDF). "이 값 이하인 데이터가 전체의 몇 %인지" 확인할 때 유용
- **rugplot**: x축 위에 데이터 하나하나를 짧은 선으로 표시. 보통 다른 플롯에 겹쳐서 실제 데이터 밀집도를 보조적으로 보여줌

### catplot (범주형, categorical)

**범주(카테고리)별로** 값을 비교할 때 사용해요.

- **stripplot**: 범주별로 데이터 포인트를 점으로 흩뿌려서 표현 (약간의 지터 적용)
- **swarmplot**: stripplot과 비슷하지만 점들이 겹치지 않게 자동 배치. 분포 모양이 더 잘 보임
- **boxplot**: 사분위수(중앙값, Q1, Q3), 이상치를 상자 형태로 요약. 그룹 간 비교에 표준적으로 많이 씀
- **violinplot**: boxplot + kdeplot을 합친 형태. 분포의 밀도까지 함께 보여줌
- **pointplot**: 범주별 평균(또는 다른 통계량)과 신뢰구간을 점과 선으로 연결. 그룹 간 추세 비교에 유용
- **barplot**: 범주별 평균값을 막대로 표현. 가장 직관적이지만 분포 정보는 안 보임

![[0902_6.png]]

petal_length 와 petal_length의 관계가 의미가 있음

#### pandas.plot

그래프 스타

[Chart visualization — pandas 3.0.5 documentation](https://pandas.pydata.org/docs/user_guide/visualization.html#visualization-other)


# 데이터분석

수업자료

![[0902_7.png]]

ML > DL

지도/비지도/강화학습

확률, 통계, 선형대수 개념을 알면 좋음

### 지도학습

- 지도 : 훈련 데이터로부터 `하나의 함수`를 유추
- 명시적인 정답이 주어진 상황에서 기계를 학습(데이터, 레이블 사용)
- 지도학습의 성능을 평가하기 위해 혼동행렬 등 인공지능 평가모델 및 교차검증 사용가능

### 지도학습의 기법

- 분류 : 이진, 다중
- 회귀: 독립변수 기반 분석, 종속변수 기반 분석(`평균으로의 회귀`. ) - 값
- 기법에 따라서 모델이 다름

통계에서 데이터는 정규분포

### 지도학습을 이용한 알고리즘

- 인공신경망 측면 : `CNN(영상)`, `RNN(언어)`
- 백터 기반 측면 : SVN , 회귀분석(Regression)

기계학습 → 지도학습 → 분류 → 신경망을 사용하면 딥러닝, 안하면 머신러닝

데이터 분석의 목적 : 군집, 분류, 규칙(pattern), 예측(Predition)

군집 VS 분류

분류(classification) - 지도 : Rnadom Forest, 의사결정트리

군집(clustering) - 비지도 —> 스펨메일, 보이스피싱 : 알고리즘 k-mean