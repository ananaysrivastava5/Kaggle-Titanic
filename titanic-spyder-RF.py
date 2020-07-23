import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dt = pd.read_csv("train.csv")
ds = pd.read_csv("test.csv")
d = pd.concat([dt,ds],ignore_index=True,sort=False)


d['Embarked'].fillna(value= d['Embarked'].mode()[0], inplace=True)

d['Cabin'].fillna(value="NA", inplace=True)

d['Salutation'] = d['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

g = d.groupby(['Sex','Pclass'])
d['Age'] = g['Age'].apply(lambda x: x.fillna(x.median()))


d["Age Range"] = pd.cut(d["Age"],[0,10,20,30,40,50,60,70,80])

d["SibSp Range"] = pd.cut(d["SibSp"],[0,1,2,3,4,5,6,7,8], include_lowest=True)

d["Parents or Children Range"] = pd.cut(d["Parch"],[0,1,2,3,4,5,6], include_lowest=True)

d['Family'] = d['Parch'] + d['SibSp']
d['Is_Alone'] = (d['Family'] == 0)

d['Fare Range'] = pd.cut(d['Fare'], [0,7.90,14.45,31.28,120], labels=["Low", "Medium", "High","Very High"], include_lowest=True)



from sklearn.preprocessing import LabelEncoder
d['Sex']= LabelEncoder().fit_transform(d['Sex'])
d['Is_Alone']= LabelEncoder().fit_transform(d['Is_Alone'])


d = pd.concat([d, pd.get_dummies(d['Pclass'], prefix='Class', drop_first=True), 
               pd.get_dummies(d['Age Range'], prefix='Age_R', drop_first=True), 
               pd.get_dummies(d['Cabin'], prefix='Cab', drop_first=True), 
               pd.get_dummies(d['Embarked'], prefix='Emb', drop_first=True), 
               pd.get_dummies(d['Salutation'], prefix='Title', drop_first=True), 
               pd.get_dummies(d['Fare Range'], prefix='Fare_R', drop_first=True)], axis=1)

d.drop(['Pclass', 'Fare','Cabin', 'Fare Range','Name','Salutation', 'Ticket','Embarked', 'Age Range', 'SibSp', 
         'Parch', 'Age','SibSp Range', 'Parents or Children Range'], axis=1, inplace=True)

# test set
X_test = d[d['Survived'].isnull()]
X_test = X_test.drop(['Survived'], axis=1)

# X set
X = d[d['Survived'].notnull()]
y = d[d['Survived'].notnull()]['Survived']
X = X.drop(['Survived'], axis=1)

# Dividing X set into training set and CV set
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 1/6, random_state = 1)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=700, min_samples_split=10, 
                                    min_samples_leaf=1, max_features='auto', oob_score=True, 
                                    random_state=1, n_jobs=-1)
classifier.fit(X_train, y_train)

# Predicting the CV set results 
y_cv_pred = classifier.predict(X_cv)
 
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_cv, y_cv_pred)

# Accuracy on CV set= 79.19

# Predicting the test set results 
y_test_pred = classifier.predict(X_test)

# Submission file making
submission = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': y_test_pred})
submission['Survived'] = submission['Survived'].astype(int)
fname= 'Titanic-mypreds-RF.csv'
submission.to_csv(fname, index=False)