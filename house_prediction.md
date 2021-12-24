# Kaggle : House Prices - Advanced Regression Techniques

![image](https://user-images.githubusercontent.com/40455392/147349209-5beee79d-155f-4e20-aa52-e95fe7d5c553.png)

![image](https://user-images.githubusercontent.com/40455392/147350590-241e43da-c8c0-48b9-8351-7cfcf8d51f75.png)

- 프로젝트 링크 : [link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard)
- 내 코드 링크 : [link](https://www.kaggle.com/seungtaekim/house-price-advanced-regression)



## 1. 프로젝트 개요

- 81개의 Columns 들을 사용해 테스트 데이터의 집값을 예측하는 모델을 만들어내는 캐글 기본 과제 중 하나입니다.
- Metric은 RSME를 사용했으며, 이미 많은 사람들에게 잘 알려진 대회이기 때문에 만점자 (0.0000)도 존재하는 대회입니다.

- 집값 예측 모델은 누구나 궁금하고, 집 가격과 관련된 다양한 조건들을 반영해야하기 때문에 여러 Feature 데이터를 만져볼 수 있는 게 장점입니다.

```
Practice Skills
- Creative feature engineering 
- Advanced regression techniques like random forest and gradient boosting
```

- 이미 프로젝트 설명란에 포함되어 있지만, Feature Engineering과 Gradient Boosting을 사용하면 좋다는 언급이 있습니다.
- 이에 따라 2021.12.23일에 수행했던 [데이콘 심장질환 예측 경진대회](https://github.com/RyanKor/dacon-heart-disease)에서 다룬 EDA Base Code를 여기서 활용해 볼 수 있을 것이란 생각에 주저하지 않고 프로젝트를 수행했습니다.



## 2. 접근 전략

- Feature가 81개나 되기 때문에 Target 데이터인 `SalePrice`와 높은 상관 관계를 갖는 데이터를 추리기 위해 Heap Map 그래프를 사용했습니다.

![image](https://user-images.githubusercontent.com/40455392/147349617-4c2bdfff-ff69-4bfd-bc62-985d3fa3e6c0.png)

- 위 상관 관계를 보면 약 11개 정도의 Feature들만 직접적으로 판매 가격과 연관성이 높은 것을 볼 수 있었고, 모델 훈련 시에 사용했습니다.

- 그러나 이 데이터에서 Integer, Float 타입의 정보만 추출해올 수 있을 뿐, Object 타입의 데이터는 반영이 안되어 있기 때문에 추후에 정수형으로 인코딩을 진행해 상관 관계를 업데이트할 예정입니다.

- 의미 있는 Feature를 하나하나 분류하는 것은 정말 어려운 일입니다. 하지만, Heap Map을 적절하게 사용하면, Feature Engineering을 하는 것에 많은 시간을 절약할 수 있습니다.



## 3. XGBoost Parameter 이해하기

```python
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:linear',colsample_bytree = 0.4603, learning_rate = 0.06, min_child_weight = 1.8,
                                 max_depth= 3, subsample = 0.52, n_estimators = 2000,
                                 random_state= 7, ntrhead = -1)
xg_reg.fit(x_train,y_train)
```

- 사용한 값은 위와 같으며, 각 파라미터별 설명은 아래와 같습니다.

```
max_depth : 트리 당 최대 깊이. 더 깊은 트리는 성능을 향상시킬 수 있지만 복잡성과 과적 합 가능성도 증가시킵니다. 값은 0보다 큰 정수 여야합니다. 기본값은 6입니다.

learning_rate : 학습률은 모델이 목표를 향해 최적화하는 동안 각 반복에서 단계 크기를 결정합니다. 학습률이 낮 으면 계산 속도가 느려지고 학습률이 높은 모델과 동일한 잔차 오류 감소를 달성하려면 더 많은 라운드가 필요합니다. 그러나 최상의 최적에 도달 할 수있는 기회를 최적화합니다. 값은 0과 1 사이 여야합니다. 기본값은 0.3입니다.

n_estimators : 앙상블의 나무 수. 부스팅 라운드 수에 해당합니다. 값은 0보다 큰 정수 여야합니다. 기본값은 100입니다.

NB : 표준 라이브러리에서는 num_round 라고합니다.

colsample_bytree : 각 트리에 대해 무작위로 샘플링 할 열의 비율을 나타냅니다. 과적 합을 개선 할 수 있습니다. 값은 0과 1 사이 여야합니다. 기본값은 1입니다.

subsample : 각 트리에 대해 샘플링 할 관측치의 비율을 나타냅니다. 낮은 값은 과적 합을 방지하지만 과소 적합으로 이어질 수 있습니다.값은 0과 1 사이 여야합니다. 기본값은 1입니다.

alpha (reg_alpha) : 가중치에 대한 L1 정규화 (올가미 회귀). 많은 기능으로 작업 할 때 속도 성능이 향상 될 수 있습니다. 정수가 될 수 있습니다. 기본값은 0입니다.

lambda (reg_lambda) : 가중치에 대한 L2 정규화 (Ridge Regression). 과적 합을 줄이는 데 도움이 될 수 있습니다. 정수가 될 수 있습니다. 기본값은 1입니다.

gamma : 감마는 의사 정규화 매개 변수 (라그랑주 승수)이며 다른 매개 변수에 따라 다릅니다. 감마가 높을수록 정규화가 높아집니다. 정수가 될 수 있습니다. 기본값은 0입니다.
```

- 가장 일반적으로 많이 이용하는 파라미터 값들에 대한 설명입니다.

- XGBoost의 장점은 회귀 및 분류 모델 모두에 좋은 성능을 가지고 있다는 것이고, L1 & L2 규제를 모두 사용하는 것이 가능합니다.
- 아직 앙상블 기법을 사용해 본 것은 아니지만, 앙상블 기법에도 XGBoost를 많이 사용한다고 하니, 다음 프로젝트에서는 앙상블과 함께 XGBoost를 사용해봐야겠습니다.



## 4. 의의

- 이제 XGBoost 모듈은 충분히 사용해봤으니, 이 프로젝트에서 상위 1%에 들었던 사람들의 코드를 보면서 다른 사람들이 더 많이 쓰는 모델은 또 무엇이 있는지 탐색해봐야 합니다.
- 기본 프로젝트지만, 타인의 프로젝트 코드를 참고하지 않았습니다. 온전히 제 스스로 처음부터 끝까지 빌드했으며, 심장 질환 예측 프로젝트 덕분에 점진적 개선 방법을 이해하고 받아들이는 과정이 즐거웠습니다.
- 학교 과제로 RSME 측정 결과물 제출은 해봤지만, 대회에서는 처음이었습니다.
- Metrics 기준이 `가격을 예측` 하는 것이기 때문에 단순 분류 모델이 아니라는 점이라 RSME를 이해하는 과정이라 생각하게 되었습니다.
- 초기에 집값 예측 프로젝트를 봤을 때, 한숨부터 나왔습니다.
  - Columns이 너무 많았고, 어떤 데이터가 의미 있는지 상관 관계를 한 번에 가져오는 방법을 생각하지 못했기 때문입니다.
- 그러나 비어있는 value를 확인하고, feature map을 뽑는 것은 이 프로젝트 수행에 있어 Columns 간의 상관관계를 파악하는 것에 시간을 아끼게 해줬기 때문에 빠른 이해를 하게 도왔던 것 같습니다.