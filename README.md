# APTOS2019
No.47 (top2%) Solution for Kaggle APTOS 2019 Blindness Detection

[inference code](https://www.kaggle.com/mikelkl/no-47-top2-solution-stacking-inference)
[detailed summary](https://zhuanlan.zhihu.com/p/81695773)

# General
This is a not bad solution to get top2% without TTA or coefficient optimization.

# Our Solution
## Data Augumentation
-  Introduce [2015 Diabetic Retinopathy competition data](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized)
- Conduct regular transformations that create less black padding
  - do_flip
  - flip_vert
  - max_zoom
## Preprocessing
- Thanks to the [@Neuron Engineer](https://www.kaggle.com/ratthachat), we refer to his [APTOS [UpdatedV14] Preprocessing- Ben's & Cropping](https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping), and set `sigmaX=10`
## Pretrained Model
- We choose [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) as our base model, this series model are quite accurate and fast to train.
## Training
- Because this is a ordinal classification task, we train it as regression problem.
- We first pretrain model on 2015 data, then finetune on 2019 data
## Ensemble
### Stage 1
- Train `efficientnet-b3, efficientnet-b4, efficientnet-b5` models on splitted 5-fold data resulting in 15 base models.
### Stage 2
- Train [xgboost](https://github.com/dmlc/xgboost), [svr](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), [catboost](https://github.com/catboost/catboost) models on logits output of stage 1 base model.
### Stage 3
- Bagging from stage 2 models
