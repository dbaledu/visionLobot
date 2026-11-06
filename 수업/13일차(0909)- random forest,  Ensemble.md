


## 수업목표
- ㅇㅇ
- 

## 실습파일
- ㅇㅇ
- 

#### 전날 수업 정리

- 랜덤포레스트
    - 불순도 : criterion ='entropy', # default='gini
    - n_estimators : 10 ~ 100 사이의 트리
    - random_state=1234 : 각 트리를 만들 때 학습 데이터를 같은 패턴으로 추

```python
rfc = RandomForestClassifier(criterion ='entropy',   # default='gini
                             n_estimators=100,       # 트리 개수   8  ->  1  -> 100 로 변경해서 해보면  
                                                     #1: versicolor,virginica,   8: virginica,setosa,  100: versicolor,setosa,
                             random_state=1234,      # 없으면 각 트리를 만들 때 학습 데이터를 무작위로 뽑는 Bootstrap Sampling
                             n_jobs= -1)
```

### 오늘 수업

- Random forest **`교차 검증 및 파라미터 최적화`**
    - **실습파일: py0909_randomforest.ipynb**
    - `GridSearchCV`
    - 하이퍼파라미터 - 사람이 학습 전에 결정하며 모델의 학습 방법을 조절. 예측값에 큰영향을 미치는
        - 파라미터의 값들을 모두 조합해서 테스트하여 최적의 값을 찾음
        - 명시하지 않으면 최적의 값을 알 수가 없음
        - 이전 테스트에 사용된 파라미터를 참조하여 지정
        - max_features : feature수에 sqrt 또는 log2 함수 적용한 결과
    - 런타임 유형 GPU L4 지정
- Boosting
    - **실습파일: py0909_2_boosting_brestCancel.ipynb**
    - Bagged와 비교
        - Random Forest는 Tree들이 비교적 독립적으로 학습하나 반면 XGBoost는 앞의 모델을 참고하여 다음 모델이 오류를 줄여갑니다.
    - 신경망과 유사
    - 모델 : XGBClassifier
        - **XGBoost(eXtreme Gradient Boosting)**에서 제공하는 **분류(Classification) 모델**
    - 이전 예측값과 가중치를 다음 학습으로 넘김 : y1에서 예측이 잘못된x 중의값들에 가중치를 반영해서 model m2에입력
- 경사하강법
    - 미분 적용
    - loss를 줄여나감. 가중치 최저점 찾기
    - 변화량이 작으면 시간이 많이 걸림. 변화량이 많으면 오차율이 많을 수 있음
- XGBoost 하이퍼 파라미터 최적화
    - 실습파일: **py0909_3_boost_최적화_brestCancel.ipynb**

머신러닝은 빠르게 최적화, 가볍다

#### 교차 검증 및 파라미터 최적화 작업순서

1. 교차검증할 파라미터 지정
2. GridSearchCV에 RandomForestClassifier 모델과 파라미터를 지정
3. fit(모델학습)을 실행하여 최적의 하이퍼파라미터 선정
4. 실제 측정된 값을 사용하여 모델 생성 및 학습
5. feature 중요도 수치 확인하고 시각화

#### 교차검증 및 파라미터 최적화

```python
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators' : [50,80,100,150,200],  # 트리 갯수
    'max_depth': [3,5,7,9,15],             # 트리 최대 깊이
    'min_samples_leaf': [8,12,15,18,20],   # 리트 노드 수
    'max_features': ['aqrt', 'log2'],      # 'auto' 제거. 노드에서 최적의 분할을 찾을 때 사용되는 feature 개수
    'min_samples_split': [5,8,10,15,20]    # 전체 데이터중에서 최소 몇개의 데이터로 트리를 생성할 것인가?                  
}
```

#### 설정한 파라미터들을 사용하여 하이퍼 파라미터를 구함

```python
rf_clf = RandomForestClassifier(n_jobs=1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=5, n_jobs=-1)
grid_cv.fit(X_train, y_train)

print('최고 하이퍼파라미터:', grid_cv.best_params_)
print('최고 예측 정확도:', grid_cv.best_score_)
```

결과

```
# 최고 하이퍼파라미터
n_estimators=50,
max_depth=9,
max_features = 'log2',
min_samples_leaf = 12,
min_samples_split= 8

# 최고 예측 정확도: 
0.9523809523809523
```

#### 하이퍼 파라미터를 사용하여 랜덤포레스트 모델 생성

```python
# 실제 측정된 값을 사용하여 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=50,
                               max_depth=9,
                              max_features = 'log2',
                              min_samples_leaf = 12,
                              min_samples_split= 8
                               )
model.fit(X_train, y_train)                              
```

#### 중요도 수치

```python
# 중요도 수치
from sklearn import metrics
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
feature_imp = pd.Series(model.feature_importances_, 
                        index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)
```

결과

```
Accuracy: 0.9555555555555556
petal width (cm)     0.444482
petal length (cm)    0.413767
sepal length (cm)    0.129884
sepal width (cm)     0.011867
dtype: float64
```

- 결과 분석 : petal(꽃받침)중요하고 sepal 의 영향은 미비함
- 중요도 체크하여 불필요한 특성을 제거 → 시각화

#### 특성 중요도 수치를 시각화

```python
# 특성 중요도 수치를 시각화
import matplotlib.pyplot as plt
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.tight_layout()
plt.show()
```


![[0909_1.png]]
#### learning_rate

- 0.01 보통 지정 : 0.1 → 0.001

### XGBoost 하이퍼 파라미터 최적화

실습파일 : py0909_3_XGBoost_앙상블_최적화_brestCancel.ipynb

#### 작업순서

1. 기본값으로 XGBClassifier 모델 생성
2. 최적의 모델 성능을 찾기 위한 하이퍼 파라미터 테스트할 리스트 지정
3. GridSearchCV 로 fit
4. 시각화

```
# 최적화된 결과
Accuracy:  0.9590643274853801
ROC_AUC:  0.9953271028037383

Confusion Matrix: 
 [[ 59   5]
 [  2 105]]

Classification Report: 
               precision    recall  f1-score   support

           0       0.97      0.92      0.94        64
           1       0.95      0.98      0.97       107

    accuracy                           0.96       171
   macro avg       0.96      0.95      0.96       171
weighted avg       0.96      0.96      0.96       171

```

```
최적화 하기전 결과
Accuracy:  0.9707602339181286
ROC_AUC:  0.9951810747663551

Confusion Matrix: 
 [[ 60   4]
 [  1 106]]

Classification Report: 
               precision    recall  f1-score   support

           0       0.98      0.94      0.96        64
           1       0.96      0.99      0.98       107

    accuracy                           0.97       171
   macro avg       0.97      0.96      0.97       171
weighted avg       0.97      0.97      0.97       171
```

참고

sklearn.datasets

![[0909_2.png|306]]


|데이터셋|함수|문제 유형|예측 대상|
|---|---|---|---|
|Iris|`load_iris()`|🌸 분류|붓꽃 품종 3종|
|Breast Cancer|`load_breast_cancer()`|🩺 분류|유방암 악성/양성|
|Wine|`load_wine()`|🍷 분류|와인 종류 3종|
|Digits|`load_digits()`|✍️ 분류|손글씨 숫자 0~9|
|Diabetes|`load_diabetes()`|📈 회귀|당뇨병 진행 정도|
|Linnerud|`load_linnerud()`|📈 다중 회귀|신체 측정값|

### XGBoost 하이퍼 파라미터 최적화
