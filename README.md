# **Turkish Music Classification**
Source : [Turkish-Music-Emotion](https://archive.ics.uci.edu/dataset/862/turkish+music+emotion)

## Objective:
Classification of verbal and nonverbal music from different genres of Turkish music into 4 discrete classes based on its Emotions,
namely: happy, sad, angry, relax.

## Dataset:
Dataset is created by extracting the intrinstic characteristics such as Mel Frequency Cepstral Coefficients (MFCCs), Tempo, Chromagram, Spectral and Harmonic features of Turkish-Music of various generes. The Dataset consists of 400 instances and 50 Fearures. The target Feature has 4 classes: happy, sad, angry, relax. 

Link : [Acoustic_features.csv](Acoustic_features.csv)

## Solution
### 1 Data Pre-processing
#### 1.1 Duplicate Removal
Duplicate instances are unnecessary and they might create bias in classification. Out of 400 instances, 12 instances were found duplicated and removed.

#### 1.2 Numerical Encoding-target feature.
Encoding classes of target feature for Model understanding and easy representation of classes. Used `LabelEncoder` from `scikit-learn` as there is no intrinstic relationship among instances of the target feature.

### 1.3 Outlier Detection & Capping.
Outliers are the points far away from the mean, and they distort it. In this Dataset, outliers are detected and capped using 3*Standard Deviation Method as the dataset is normal, except a few features(skew 1-3).
After capping outliers, some of the features became close to normal distribution(skew b/w [-1.5 1.5]).








