[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/g6ZC_OOE)
# House Price Prediction | 아파트 실거래가 예측

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | 
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김영천](https://github.com/dudcjs2779)             |            [배창현](https://github.com/Bae-ChangHyun)             |            [조예람](https://github.com/huB-ram)             |
|                            팀장                             |                            팀원                             |                            팀원                             |

## 1. Competetion Info

### 1-1 Overview

House Price Prediction 경진대회는 주어진 데이터를 활용하여 서울의 아파트 실거래가를 효과적으로 예측하는 모델을 개발하는 대회입니다. 

부동산 실거래가의 예측은 시세를 예측하여 적정한 가격에 구매와 판매를 도와주게 합니다. 그리고, 정부의 입장에서는 비정상적으로 시세가 이상한 부분을 체크하여 이상 신호를 파악하거나, 업거래 다운거래 등 부정한 거래를 하는 사람들을 잡아낼 수도 있습니다. 

저희는 이러한 목적 하에서 다양한 부동산 관련 의사결정을 돕고자 하는 부동산 실거래가를 예측하는 모델을 개발하는 것입니다. 특히, 가장 중요한 서울시로 한정해서 서울시의 아파트 가격을 예측하려고합니다.

### 1-2 Timeline

Jan 15, 2024 ~ Jan 25, 2024

### 1-3 Evaluation

RMSE

해당 시점의 매매 실거래가를 예측하는 Regression 대회이며, 평가지표는 RMSE(Root Mean Squared Error)를 사용합니다.
RMSE는 예측된 값과 실제 값 간의 평균편차를 측정합니다. 아파트 매매의 맥락에서는 회귀 모델이 실제 거래 가격의 차이를 얼마나 잘 잡아내는지 측정합니다. 

## 2. Components

### Directory
<pre>
│  통합본.ipynb
├─code
│  │  gbm_final01.pkl
│  │  lightgbm_sklearn_api.pkl
│  │  output.csv
│  │  predict_apart.ipynb
│  │  predict_apart_EDA.ipynb
│  │  
│  └─data
│          base_rate.csv
│          basic_apart.parquet
│          filled_loc.csv
│          test.csv
│          train.csv
│          
└─
</pre>      

## 3. Data descrption

### 3-1 Dataset overview

Train: 2007년 1월 1일부터 2023년 6월 30일 사이의 예측해야 할 거래금액(target)을 포함한, 52개의 아파트의 정보에 대한 변수와 거래시점에 대한 변수 총 1,118,822개

Test: 학습 데이터기간 이후 3개월인 2023년 7월 1일부터 2023년 9월 26일까지의 정보로 구성되어 총 9272개

base_rate: [한국은행경제통계시스템](https://ecos.bok.or.kr/#/)에서 가져온 금리 데이터입니다.

basic_apart: [K-apt 공동주택관리정보시스템](https://www.k-apt.go.kr/board/boardList.do?board_type=03)에서 가져온 아파트 단지 정보입니다.

filled_loc: 해당 데이터는 프로젝트 진행당시 팀원분께서 naver API Geocode 를 이용해서 만든 위경도 데이터입니다.


### 3-2 EDA

#### 결측치
결측치는 아파트 단지정보에대한 데이터인 base_rate.csv 파일을 통해서 train 데이터와 중복되는 컬럼으로 결측치를 채워주었습니다.

#### 상관관계
<img src="https://github.com/dudcjs2779/seoul-apartment-prices-prediction/assets/42354230/e0400b5a-8c48-441c-964a-25f370ec63b8" alt="corr_image" width="800">

위 그림은 제 모델에서 target값을 예측하는데 사용한 주요한 변수들의 피어슨 상관관계의 heatmap 입니다.<br>
위 피쳐들에는 train, 외부데이터 및 Feature engineering 해서 생성한 파생변수까지 포함돼 있습니다.

기본적으로 train 데이터의 피쳐들 중에서 target값과 상관관계가 높은 피쳐들 위주로 EDA 후 파생변수를 생성하였고 개인적으로 target 값과의 상관관계는 높지 않으나 유의미할 거라 생각되는 피쳐들 또한 실험을 통해 모델 성능과 Feature importance를 확인해가며 학습에 활용하였습니다.

자세한 EDA 내용은 predict_apart_EDA.ipynb 의 EDA 파트를 참고해주세요.


### 3-3 Feature engineering
- 건물나이
  - 계약일 기준 건물의 신축 및 구축의 여부를 간접적으로 나타내기 위한 피쳐로 계약년월일과 건축년도의 차이를 일수(days)로 나타낸 컬럼입니다.
- 계약경과일
  - 계약년월일이 최근에 가까울 수록 거래가가 높아지는 현상을 더욱 잘 대변할 수 있도록 현재 날짜와 계약년월일의 차이를 구한 컬럼입니다.
- 전용면적_mean
  - 시군구 + 아파트명으로 groupby해서 해당 아파트의 전용면적의 평균값을 구해 해당 아파트의 전용면적이 크거나 작은 아파트의 비율을 대략적으로 나타내기 위해 생성한 컬럼입니다.
- 범주4비율
  - 시군구 + 아파트명으로 groupby해서 (전용면적 범주 대형에 포함되는 세대수 / 해당 아파트의 전체 거래횟수)를 구해 대략적인 해당 아파트의 전용면적이 큰 아파트가 많은지에 대한 비율을 나타냅니다.
  - 전용면적이 상당히 높은 세대가 포함된 아파트는 대체로 전용면적이 높은 세대의 비율이 높아 고급 아파트를 구별하는데 도움이 될 것이라 판단했습니다.
- 3월평단가_std
  - 3개월단위로 나타내는 Datetime 컬람을 생성하고 시군구 + 아파트명으로 groupby해서 3개월간의 평단가의 분산을 나타내는 컬럼을 생성했습니다.
  - 가격의 변동이 높은 아파트들이 계속해서 가격의 변동이 심할 것이라는 가설에 기반합니다.
  - 가격이 높은 아파트들이 대체로 분산이 높아 높은 가격의 아파트를 구분하는데 도움이 될 것이라 판단했습니다.
- 세대당주차대수
  - 주차대수와 세대수 컬럼을 이용해서 세대당 주차대수 컬럼 생성했습니다.
  - 세대당주차대수가 주차대수 컬럼에 비해서 target 값과 상관관계가 약 0.16정도 높게 나왔습니다.
- 고층도
  - 층과 최고층수 컬럼을 이용해서 고층도(해당 아파트에서 고층인 정도의 비율) 컬럼을 생성했습니다.
- 동순위
  - 동별 평균 평단가를 구한 뒤 순위를 매겨 동별로 평단가가 컬럼을 대변하도록 의도했습니다.
- 아파트순위
  - 시군구+아파트명별로 평균 평단가를 구한 뒤 순위를 매겨 아파트별로 평단가가 컬럼을 대변하도록 의도했습니다.
- 이전계약_diff
  - 시군구 + 아파트명별로 이전계약과 현재계약의 차이를 나타내는 컬럼을 생성했습니다.
  - 부동산 거래 데이터 특성상 이전 거래에 영향을 많이 받기 때문에 이전 거래와의 일수의 차이로 간접적으로 이전 거래의 영향을 나타낼 수 있다고 생각했습니다.

## 4. Modeling

### 4-1 Model descrition
- LGBM

LGBM으로 모델링을 진행했습니다. 다수의 결측치가 포함돼 있어 nan값에 강경한 트리모델을 사용해야된다고 생각했으며 110만개 이상의 데이터를 학습하기에는 LGBM을 사용하는 게 좋을거라 생각했습니다. 만약에 옛날의 거래내역 데이터가 최근의 데이터를 예측하는데 방해가 된다면 판단한다면 예전의 데이터는 제거해서 데이터의 갯수를 줄이고 XGBoost나 Catboost를 사용하는 것도 좋을 것 같습니다.

해당 모델은 기본적으로 holdout 방식으로 성능 검증을 진행합니다. 부동산 데이터도 이전 거래내역이 다음 거래내역에 영향을 미치는 일종의 시계열 데이터의 성향이 있기 때문에 검증 데이터셋을 최근 데이터로 설정할 필요가 있다고 판단했습니다.

holdout 방식은 검증 데이터를 학습에 사용할 수 없지만 부동산 데이터의 특성상 최근 데이터가 굉장히 중요하기 때문에 이를 보안하기 위해 oputna로 하이퍼파라미터 튜닝후 해당 하이퍼파라미터를 그대로 사용해서 검증데이터를 포함해 모든 데이터를 학습했습니다.

#### 평가
![02](https://github.com/dudcjs2779/seoul-apartment-prices-prediction/assets/42354230/ebf81a15-1de0-4051-9fbb-80ce86d300d5)
- RMSE: 12840.8993


### 4-2 Modeling Process

<img width="465" alt="스크린샷 2024-01-26 오전 1 17 49" src="https://github.com/UpstageAILab/upstage-ml-regression-04/assets/120548753/07a753ac-66f0-4578-b8a3-85164d5cd958">


## 5. Result

### 5-1 Leader Board
#### 대회 시스템상 이슈로 재채점이 이루어졌습니다.
- ~Rank:9~
- ~Score:106512.0996~

![스크린샷 2024-01-29 142148](https://github.com/dudcjs2779/seoul-apartment-prices-prediction/assets/42354230/919a779f-0f2a-4e76-8235-6a70e9b9d483)
- Rank:2
- Score:10764.6959

### 5-2 Presentation

- docs/pdf 폴더 참조

### Reference
- 실거래지수: KOSIS국가통계포털(https://kosis.kr/statHtml/statHtml.do?orgId=408&tblId=DT_KAB_11672_S1)
- 서울시 학교: 서울열린데이터광장(https://data.seoul.go.kr/dataList/OA-20502/S/1/datasetView.do)
- 아파트 정보: K-apt 공동주택관리정보시스템(https://www.k-apt.go.kr/board/boardList.do?board_type=03)
- 금리: 한국은행경제통계시스템(https://ecos.bok.or.kr/#/)
- 가구총소득: 서울열린데이터광장(https://data.seoul.go.kr/dataList/DT201013B022/S/2/datasetView.do)
