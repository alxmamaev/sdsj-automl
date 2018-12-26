![](https://github.com/alxmamaev/image-storage/blob/master/sdsj2018/Снимок%20экрана%202018-11-04%20в%2017.17.38.png)
# Sberbank AutoML solution

## Dataset preparation
* If the dataset is big (>2GB) then we calculate features correlation matrix and the delete correlated features
* Else we make Mean Target Encoding and One Hot Encoding. 
* After that, we select top-10 features by coefficients of the linear model (Ridge/LogisticRegression)
* We generate new features by pair division from top-10 features. This method generates 90 new features (10^2–10) and concatenates it to the dataset.


## Model training
* If the dataset is small then we can train three LightGBM models by k-folds, after that blend prediction from every fold.
* If the dataset is big and the time limit is small (5 minutes) then we just train linear models (logistic regression or ridge)
* Else we train one big LightGBM (n_estimators=800)

## Result
5th place on private leaderboard
