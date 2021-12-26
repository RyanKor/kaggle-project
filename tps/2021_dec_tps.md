# Tabular Playground Series : Multi-class Classification

![image](https://user-images.githubusercontent.com/40455392/147399352-02585c4e-6605-4a0f-9889-9bf608496bbd.png)



![image](https://user-images.githubusercontent.com/40455392/147412809-a58d435e-1417-49e5-a92a-c5600072f503.png)

- Project Final Score : 0.95203
- 프로젝트 링크 : [link](https://www.kaggle.com/c/forest-cover-type-prediction/overview)
- 제출 코드 링크 : [link](https://www.kaggle.com/seungtaekim/tps-lgbm-baseline)



## 1. 프로젝트 개요

- 캐글에서 매달 진행하는 TPS 시리즈의 2021년 마지막 경진대회입니다.
- 다중 분류 모델 자료를 제공하고 있으며, 이미지 분류 모델은 아닙니다.
  - 주어진 정수형 데이터에 기반한 클래스 분류 모델
- 별 생각없이 참여했지만, 프로젝트 구성이 워낙 단순한 탓에 빠른 제출 및 모델 성능을 평가할 수 있었습니다.



## 2. 접근 전략

- 간단한 EDA를 진행했고, 클래스별 불균형이 매우 심각했습니다.

![image](https://user-images.githubusercontent.com/40455392/147399406-6eb9fbb0-7ccd-4b24-8852-df7490fb6f68.png)

- 주어진 데이터들의 각 row가 이미지 데이터에 대한 정보이기 때문에 Id 칼럼을 제외한 특정 Feature를 소거하고 학습하지 않았습니다.
- 다만, 위의 이미지에서 보는 것처럼 3 ~ 7 클래스의 데이터는 거의 없다시피한 상황입니다.
- 모델 성능이 99%의 정확도를 보여도 1%에서 3 ~ 7 클래스에 대해 오답을 만든다면, 특정 데이터만을 분류하는 상황이기 때문에 좋은 모델이라고 할 수 없습니다.
  - 즉, 현재 Data Skewed 상황이기 때문에 이에 대한 조치가 필요했습니다.
- 데이터 포맷이 Structured Data이기 때문에 기존에 잘 알고 활용했던 Unstructured Data에 대한 Data Augmentation을 수행하는 것은 어렵다고 판단했고, Fold를 나눠 학습하는 방식을 처음에는 떠올렸습니다.
- 그러나 각 클래스별 훈련 데이터의 수를 확인하고 나서, Fold를 이용한 Cross-Validation을 사용하기에는 5번 클래스의 데이터가 거의 없다시피한 것을 확인할 수 있었습니다.

```python
train_df["Cover_Type"].value_counts()

# 아래는 결과
2    2262087
1    1468136
3     195712
7      62261
6      11426
4        377
5          1
```



(참고) TPS의 훈련/테스트 데이터 모두 missing value는 존재하지 않았습니다.

```python
check_missing_col(train_df)
check_missing_col(test_df)

# no missing cols
```



- 교차 검증 (Cross validation)의 경우 다음과 같은 장점 때문에 많이 활용되는 것으로 알고 있습니다.
  - 모든 데이터셋을 평가에 활용할 수 있다.
  - 평가에 사용되는 데이터 편중을 막을 수 있다.
  - 평가 결과에 따라 좀 더 일반화된 모델을 만들 수 있다.
- 그러나 훈련 데이터에서 특정 데이터의 클래스 명칭은 존재하지만, 데이터가 0 ~ 1개 밖에 존재하지 않는 경우 교차 검증을 사용한 학습 방법이 1개 또는 아예 학습을 하지 못하는 상황이라고 판단했기 때문에 이와 같은 모델 학습 방법은 되려 1개의 데이터에 대한 overfit이 발생할 수 있다고 판단했습니다.
- 따라서, 균형잡힌 데이터 학습에 대한 관점에서 클래스 불균형 해소를 하는 방향으로 사이킷런의 class_weight를 사용하는 것을 떠올려 바로 방법을 적용하게 되었습니다.

```python
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
# print(class_weights)
weights = {1 : 3.89057751e-01, 2 : 2.52647748e-01, 3 : 2.92446028e+00, 4 : 1.54440154e+03,
          5 : 4.57142857e+05, 6 : 4.99336818e+01, 7 : 9.17533784e+00}
```

- 위의 수를 소수점 표기 방법으로 바꾸면, 아래와 같이 5번 클래스에 대해 막중하게 가중치 조절이 된 것을 볼 수 있습니다.

```python
class_weight={1: 0.389057751, 2: 0.252647748, 
              3: 2.92446028, 4: 1544.40154, 5: 457142.857, 
              6: 49.9336818, 7: 9.17533784}
```

- 그러나 조절된 가중치 값의 편차가 크다고 놀랄 필요는 없습니다. 이는 모델을 학습하기 위해 적절하게 취해진 결과이고, 이후 학습된 모델의 정확도와 별개로 `좋은 모델` 을 구성하기 위해 컴퓨터가 적절하게 조절해 준 값이기 때문입니다.



- 클래스 가중치 불균형 해소 방안을 찾았다면, 본격적으로 수행해줘야할 것은 학습을 수행할 모델을 선별하는 일이었습니다.

  - 이를 위해 처음에는 많이 사용해왔던 XGBoost를 생각했습니다.

  - 그러나 특정 모델만을 반복적으로 사용해오는 것은 좋지 못하고, 캐글 유저들에게 많은 사랑을 받는 다른 부스팅 모델을 사용해보고 싶었습니다.

    - 이에 따라, 추가적으로 2가지 모델을 더 탐색해보게 되었습니다.
      - LGBM
      - CatBoost

  - 이 중 가장 먼저 사용한 것은 LGBM입니다.

    - 훈련 데이터의 총 row 값은 15120 개 입니다.
    - 테스트 데이터의 총 row 값은 565892 개 입니다.

    - XGBoost의 경우 찾아보니 LGBM과 성능은 비슷하지만, 더 무거운 모델이라 알려져 있고, 지금처럼 데이터가 많은 경우 학습 시에 LGBM보다 학습 속도가 더 느릴 것이라 판단했습니다.
    - (실제로 LGBM으로 n_estimator 값을 1000이라고 설정해주면 모델 학습에 40분 정도 걸리는 것을 볼 수 있습니다.)
      - TPS 경진대회라고 모델 학습 속도를 최대 20분 정도로 보고 있었는데, 경량화된 모델이라도 데이터 양을 처리하는데 시간이 필요한 것을 알 수 있었습니다)

  - 초기에는 LGBM의 별다른 하이퍼 파라미터 조정 없이 학습을 진행했고, 제출 결과 정확도가 92% 가까이 나왔습니다.

  ![image](https://user-images.githubusercontent.com/40455392/147399933-1d0fc89d-49a7-48e0-ad69-4074bf7b7794.png)

  

  - 높은 점수를 얻은 것 같지만, 아무런 매개 변수 조정 없이 이러한 점수를 얻었다는 것은 다른 사람들의 점수는 이미 월등하게 높을 것이라는 막연한 추측을 했고, 이 추측은 맞았습니다.

  ![image](https://user-images.githubusercontent.com/40455392/147399945-c2dce386-450c-48d8-b2b8-1d1f22cd2d74.png)

  

  - 1000번 학습 결과 정확도 개선 결과

  ![image](https://user-images.githubusercontent.com/40455392/147407780-663851a3-5366-4151-a320-a68b5c4773f9.png)

  

  - 4000번 학습 결과 정확도 : 1등과 정확도가 0.521% 밖에 차이가 나지 않는다.

  ![image](https://user-images.githubusercontent.com/40455392/147412809-a58d435e-1417-49e5-a92a-c5600072f503.png)

  - 첫 제출에서 이미 전체 참여자의 80% 정도 밖에 안되는 점수라는 것을 확인했고, 1등 점수를 보니 0.95724로 대부분의 점수가 3% 내외에서 결정된다는 것을 알게 되었습니다.

- 이후, LGBM의 기본 값에서 매개변수를 조정해 다시 학습을 수행했습니다.

  ```python
  from lightgbm import LGBMClassifier
  # lgbm1 -> default parameter of LGBM
  
  lgb_params = {'n_estimators'     : 4000,      # Number of boosting iterations.
                'random_state'     : 42,            # Random seed initilizer for the model, helps to replicate the experiments.
                'learning_rate'    : 0.1,              # The model learning rate.
                'subsample'        : 0.95,            # Row subsample from the dataset, like feature_fraction, but this will randomly select part of data without resampling
                'subsample_freq'   : 1,               # Use or not subsample frequency.
                'colsample_bytree' : 0.75,            # LightGBM will randomly select a subset of features on each iteration (tree).
                'reg_alpha'        : 0.5,             # L1 regularization.
                'reg_lambda'       : 0.5,             # L2 regularization.
                'min_child_weight' : 1e-3,            # Minimal sum hessian in one leaf, it can be used to deal with over-fitting.
                'min_child_samples': 32,              # Minimal number of data in one leaf. Can be used to deal with over-fitting.
                'objective'        : 'multiclass',    # Softmax objective function.
                'metric'           : 'multi_logloss', # Log loss for multi-class classification.
                'device_type'      : 'gpu',
               }   
  
  lgbm1 = LGBMClassifier(class_weight=weights, **lgb_params)
  ```

  - 부스팅 반복 횟수 (`n_estimators`)를 1000번으로 했더니, 기본 값 100회 일때보다 성능이 더욱 향상되었고, 검증 데이터 스코어가 `0.95430875`로 향상된 것을 볼 수 있었습니다.
  - `early_stopping` 을 걸어뒀으므로 부스팅 반복 횟수를 4000까지 늘리고, 손실값이 꾸준히 증가하는 지점을 찾아 최대 학습 위치에서 학습을 중단하는 과정으로 접근해야할 것으로 보입니다.
    - 4000번 진행 시, 1000번 때보다 logloss 값을 약 1.6% 개선시킬 수 있었습니다.
    - 그러나 4000번 이후에도 손실 값이 지속적으로 감소했기 때문에 n_estimator 값을 더 증가시켜 학습시키는 과정이 필요해 보입니다.
  - RMSE, F1을 metric으로 사용했을 때와는 별개로 클래스를 정확하게 분류할 수록 점수가 늘어나는 과정이기 때문에 높은 정확도가 제출 점수와 비례할 것으로 보입니다.

  ```bash
  [91]	valid_0's multi_logloss: 0.167273
  [92]	valid_0's multi_logloss: 0.166571
  [93]	valid_0's multi_logloss: 0.166122
  [94]	valid_0's multi_logloss: 0.165783
  [95]	valid_0's multi_logloss: 0.165505
  [96]	valid_0's multi_logloss: 0.164995
  [97]	valid_0's multi_logloss: 0.164699
  [98]	valid_0's multi_logloss: 0.164233
  [99]	valid_0's multi_logloss: 0.163903
  [100]	valid_0's multi_logloss: 0.163478
  ...
  [990]	valid_0's multi_logloss: 0.112506
  [991]	valid_0's multi_logloss: 0.112495
  [992]	valid_0's multi_logloss: 0.112493
  [993]	valid_0's multi_logloss: 0.112481
  [994]	valid_0's multi_logloss: 0.112467
  [995]	valid_0's multi_logloss: 0.112458
  [996]	valid_0's multi_logloss: 0.112442
  [997]	valid_0's multi_logloss: 0.112434
  [998]	valid_0's multi_logloss: 0.112426
  [999]	valid_0's multi_logloss: 0.112417
  [1000] valid_0's multi_logloss: 0.112398
  ...
  [3991]	valid_0's multi_logloss: 0.0966306
  [3992]	valid_0's multi_logloss: 0.0966358
  [3993]	valid_0's multi_logloss: 0.0966324
  [3994]	valid_0's multi_logloss: 0.0966286
  [3995]	valid_0's multi_logloss: 0.0966271
  [3996]	valid_0's multi_logloss: 0.0966198
  [3997]	valid_0's multi_logloss: 0.0966216
  [3998]	valid_0's multi_logloss: 0.0966168
  [3999]	valid_0's multi_logloss: 0.0966139
  [4000]	valid_0's multi_logloss: 0.0966106
  ```

  

## 후기

- 데이콘의 심장 질환 예측하기, 집 값 예측하기 심화, 그리고 TPS 를 지난 3일 동안 연속으로 참여했고, 3개 모두 어느 정도 좋은 결과를 예측할 수 있었습니다.
- 솔루션을 항상 딥러닝으로 제시하기 보다 머신러닝이라는 큰 틀 안에서 Data-Centric 관점에서 데이터 퀄리티를 높이거나 (Data Augmentation, Synthetic Data 등) 앙상블, 새로운 논문의 모델 성능을 테스트 해보는 등의 시도를 꾸준히 해봐야 합니다.
- 수치에 속으면 안됩니다. 훈련 데이터 속 라벨링 데이터 중 5번 카테고리는 1개 밖에 없었고, 교차 검증 등으로 학습 시키기에는 데이터가 소외될 가능성이 너무 높았습니다.
  - 다양한 관점에서 데이터를 보기 위해 EDA를 여러 관점에서 시도해봐야합니다.
- 최종 스코어는 0.95203 이지만, 1등 점수가 0.95724 이기 때문에 실제적으로 퍼센트로 따져봤을 때 0.521% 밖에 차이나지 않습니다.
  - 대회 자체가 쉽다보니, 사소한 소숫점 차이로 많이 갈립니다.
  - 소숫점에서 갈리다 보니, 점수 차이가 많이 안나는데도 상위 60% 정도밖에 되지 않는 것을 볼 수 있습니다.

- 조금씩 자신감이 올라오고 있는 것이 느껴집니다 :)