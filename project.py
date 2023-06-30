import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

file_breast_cancer = pd.read_csv("breast-cancer.csv")

"""
DIAGNOSIS
M - Malignant(BAD)
B - Bengin(GOOD)
"""

# cleaning data
file_breast_cancer["diagnosis"] = file_breast_cancer["diagnosis"].replace(to_replace="M", value=1)
file_breast_cancer["diagnosis"] = file_breast_cancer["diagnosis"].replace(to_replace="B", value=0)

clean_file_breast_cancer = file_breast_cancer

# choosing best features
"""
test = SelectKBest(score_func=chi2, k=10)

fit = test.fit(X, y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ["Specifications", "Score"]
#print(featureScores.nlargest(10, "Score"))
"""


Best_Features = ["area_worst", "area_mean", "area_se", "perimeter_worst", "perimeter_mean","radius_worst", "radius_mean", "perimeter_se", "texture_worst", "texture_mean"]

#setting X and y
X = clean_file_breast_cancer.drop(columns="diagnosis")
y = clean_file_breast_cancer["diagnosis"]

#train_test_split
train_X, value_X, train_y, value_y = train_test_split(X, y, random_state=1, test_size=0.2)

#K Nearest Neighbors Classifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(train_X, train_y)

#prediction
y_pred = knn_model.predict(value_X)


#summary
print(classification_report(value_y, y_pred))  # 0 - Bengin, 1 - Malignant
print(f"accuracy score equals: {round(accuracy_score(value_y, y_pred)*100, 2)}%")
print(f"mean absolute error equals: {round(mean_absolute_error(value_y, y_pred)*100, 2)}%")

