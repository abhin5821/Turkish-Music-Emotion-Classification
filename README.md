# **Turkish Music Classification**
Source : [Turkish-Music-Emotion](https://archive.ics.uci.edu/dataset/862/turkish+music+emotion) |  Main File : [main.ipynb](main.ipynb)

## Objective:
Classification of verbal and nonverbal music from different genres of Turkish music into 4 discrete classes based on its Emotions,
namely: happy, sad, angry, relax.

## Dataset:
Dataset is created by extracting the intrinstic characteristics such as Mel Frequency Cepstral Coefficients (MFCCs), Tempo, Chromagram, Spectral and Harmonic features of Turkish-Music of various generes. The Dataset consists of 400 instances and 50 Fearures. The target Feature has 4 classes: happy, sad, angry, relax. 

Link : [Acoustic_features.csv](Acoustic_Features.csv)

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

Results: 

  <img width="673" alt="Screenshot 2024-12-12 at 23 06 01" src="https://github.com/user-attachments/assets/3d7e7529-60d9-41c2-9fd9-fee074987f8e" />

Confusion Matrix: 

  ![16be4417-93c6-4761-9f49-b51b7f6bbb51](https://github.com/user-attachments/assets/799b35d8-0fcb-4582-bd84-f643d02a6d03)

### 4. Evaluation and Results
For Evaluation of the model, I have used Accuracy & F1-Score (Macro) as the main metric. As per the problem statement, all the classes have equal priority as this is Music Emotion Classification unlike Heart-disease, cancer dataset etc.. Other metrics include Precision-Macro & Recall-Macro. Using macro is evident that the average value is enough to justify the model's performance rather than that of individual classes'.

Here are the results from various other classifiere:

1. Training with base and ensemble classifiers with feature selection.

  <img width="686" alt="Screenshot 2024-12-12 at 23 31 25" src="https://github.com/user-attachments/assets/2620abbf-cf9f-4a5c-9c6c-97f4bf105785" />

2. Trained Random Forest Classifier with hyper-parameter tuining using RandomisedSearchCV

  >Random Forest Algorithm trained with hyper-parameter tuining gave lesser F1-score compared to Simple RF.
  
  <img width="683" alt="Screenshot 2024-12-12 at 23 37 00" src="https://github.com/user-attachments/assets/3af02212-7c07-4219-8255-28594189118d" />

3. Training using Stacking Classifiers.
   
  <img width="534" alt="Screenshot 2024-12-12 at 23 39 10" src="https://github.com/user-attachments/assets/46b72df4-a0ee-46cd-80f1-fa2039477a51" />
 
4. Training with base and ensemble classifiers without feature selection.
  
   <img width="682" alt="Screenshot 2024-12-12 at 23 40 15" src="https://github.com/user-attachments/assets/d111890a-449c-490e-8f8a-5e254186484b" />

### 5. Conclusions
The given dataset, [Acoustic_Features](Acoustic_Features.csv), based on its nature and number of features, we have applied feature engineering techniques: Mutual Information with Sequential Forward Selection(MISFS), finally given the extracted features to RandomForestClassifier, which performed best among all other classifiers with an accuracy of **84.62%** and F1-Score of **84.78%**.

### References
1. https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification
2. https://www.mdpi.com/2079-9292/12/10/2290#sec3-electronics-12-02290
3. https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification
4. https://www.evidentlyai.com/classification-metrics/multi-class-metrics
5. https://scikit-learn.org/stable/modules/preprocessing.html
6. https://medium.com/@vinodkumargr/07-standardization-and-normalization-techniques-in-machine-learning-standardscaler-3890a89bddbf.










