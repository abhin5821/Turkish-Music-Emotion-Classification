# **Turkish Music Classification**
Source : [Turkish-Music-Emotion](https://archive.ics.uci.edu/dataset/862/turkish+music+emotion) |  Main File : [main.ipynb](main.ipynb)

## Objective:
Classification of verbal and nonverbal music from different genres of Turkish music into 4 discrete classes based on its Emotions,
namely: happy, sad, angry, relax.

## Dataset:
Dataset is created by extracting the intrinstic characteristics such as Mel Frequency Cepstral Coefficients (MFCCs), Tempo, Chromagram, Spectral and Harmonic features of Turkish-Music of various generes. The Dataset consists of 400 instances and 50 Fearures. The target Feature has 4 classes: happy, sad, angry, relax. 

Link : [Acoustic_features.csv](Acoustic_features.csv)

## Solution
### 1. Data Pre-processing
#### 1.1 Duplicate Removal
Duplicate instances are unnecessary and they might create bias in classification. Out of 400 instances, 12 instances were found duplicated and removed.

#### 1.2 Numerical Encoding-target feature.
Encoding classes of target feature for Model understanding and easy representation of classes. Used `LabelEncoder` from `scikit-learn` as there is no intrinstic relationship among instances of the target feature.

#### 1.3 Outlier Detection & Capping.
Outliers are the points far away from the mean, and they distort it. In this Dataset, outliers are detected and capped using 3*Standard Deviation Method as the dataset is normal, except a few features(skew 1-3).
After capping outliers, some of the features became close to normal distribution(skew b/w [-1.5 1.5]).

### 2. Feature Engineering
#### 2.1 Feature Scaling 
Feature Scaling brings all the features to one scale. I have applied `StandardScaler` from `scikit-learn` as the features were normal.

#### 2.2 Mutual Information with Sequential Forward Selection(MISFS)
Feature Selection is crucial so as to reduce the dimensionality of the dataset and also to select relevent features to enhance the model's performance. Before selecting the optimum number of features, the dataset has been split into train and test with a ratio of 70:30 using `train-test-split` from `scikit-learn`. The feature engineering is applied only on train dataset. This method is divided in two phases:
- Phase 1: Mutual Information(Filter method)
  
This phase involves computing the importance of the feature to the target variable and arranging them in the decreasing order. Based on trail and error method, 47 features have been selected based on their imporatnce with target and among themselves. `mutual_info_classif` from `scikit-learn` is used to do the same.

* Phase 2: Sequential Forward Selection (Wrapper Method)

This phase involves Selecting the optimum features that gives the best accuracy. This method begins by selecting a single feature to train and predict against target variable, followed by pair, three and so on, until the optimum group of features gives the best performance. The scoring is based on accuracy of the group of features. Out of 50 features, 23 optimum features were selected. `SequentialFeatureSelector` from `scikit-learn` and `KNeighborsClassifier` have been used.

After selecting the optimum number of features from training dataset, the same number of features are selected from test dataset and they are given to the multiple classifiers.

### 3. Model Selection & Training.
In this Model selection and training process, the reduced dataset from previous step is given to multiple classifiers including base classifiers, ensemble classifers & Stacking Classifiers. Out of all the classifiers, `RandomForestClassifer` achieved the highest accuracy of 0.8462 and F1-Score (Macro) of 0.8478.


Confusion Matrix: 

![16be4417-93c6-4761-9f49-b51b7f6bbb51](https://github.com/user-attachments/assets/799b35d8-0fcb-4582-bd84-f643d02a6d03)










