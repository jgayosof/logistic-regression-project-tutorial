# Step 0. Load libraries and modules
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier


df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')


# Drop NAs & duplicates
df_raw = df_raw.dropna().drop_duplicates()


# 'job' analysis, Replace unknown with mode:
condition  = (df_raw['job'] == 'unknown')
df_raw.loc[condition, 'job'] = df_raw['job'].mode()

# 'marital' analysis, replace unkwnown with mode:
condition  = (df_raw['marital'] == 'unknown')
df_raw.loc[condition, 'marital'] = df_raw['marital'].mode()

# 'education' analysis, Replace unknown with mode:
condition  = (df_raw['education'] == 'unknown')
df_raw.loc[condition, 'education'] = df_raw['education'].mode()

# 'education' analysis, Insert categories 'basic.9y','basic.6y','basic4y' into 'middle_school':
df_raw['education'] = df_raw['education'].replace(['basic.9y', 'basic.6y', 'basic.4y'], 'middle_school')

# 'default' analysis, Replace unknown with mode:
condition  = (df_raw['default'] == 'unknown')
df_raw.loc[condition, 'default'] = 'no'

# 'housing' analysis, Replace unknown with mode:
condition  = (df_raw['housing'] == 'unknown')
df_raw.loc[condition, 'housing'] = df_raw['housing'].mode()

# 'loan' analysis, Replace unknown with mode:
condition  = (df_raw['loan'] == 'unknown')
df_raw.loc[condition, 'loan'] = df_raw['loan'].mode()

# 'contact' analysis, Replace unknown with mode:
condition  = (df_raw['contact'] == 'unknown')
df_raw.loc[condition, 'contact'] = df_raw['contact'].mode()

# 'age' analysis, remove 'age outliers
df_raw = df_raw.drop(df_raw[df_raw.age > 69].index)
# Convert age into categorical data by creating age-groups of ten years
df_raw['age'] = pd.cut(x=df_raw['age'], bins=[10, 20, 30, 40, 50, 60], 
                       labels=['10-20', '21-30', '31-40', '41-50', '51-60'])

# 'duration' analysis
# remove rows which duration = 0
df_raw = df_raw.drop(df_raw[df_raw.duration == 0].index)
# remove duration outliers
df_raw = df_raw.drop(df_raw[df_raw.duration > 1000].index)

# 'campaign' analysis, 'campaign' to categorical
df_raw['campaign'] = pd.Categorical(df_raw.campaign)


df_raw.to_csv('../data/raw/bank-marketing-campaign_raw.csv')
df_interim = df_raw.copy()


# drop columns due to high correlations
correlation_deletes = ['cons.price.idx', 'nr.employed', 'euribor3m']
df_interim = df_raw.drop(columns=correlation_deletes)

# Los datos están desbalanceados.
# Voy a aplicar la técnica de 'subsampling' para los casos de 'no'
# pasando de 36k a 6k:
N = 6000
# all rows w y=yes:
df_yes = df_interim.loc[df_interim['y'] == 'yes']
# all rows w y=no:
df_no = df_interim.loc[df_interim['y'] == 'no']
# N subsample w y=no:
df_no_subsampled = df_no.sample(n=N)

df_interim.to_csv('../data/interim/bank-marketing-campaign_interim.csv')
# concatenation:
df_processed = pd.concat([df_yes, df_no_subsampled])

# Data scaling:
scaler = MinMaxScaler()
columns = ['duration', 'campaign', 'pdays', 'previous','emp.var.rate','cons.conf.idx']
df_scaler = scaler.fit(df_processed[columns])
df_processed[columns] = df_scaler.transform(df_processed[columns])

df_processed = pd.get_dummies(df_processed, columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y'])
df_processed.to_csv('../data/processed/bank-marketing-campaign_processed.csv')


y = df_processed['y_yes']
X = df_processed.drop(['y_yes', 'y_no'], axis=1)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.7, random_state=13)

# Logistic Regression model and fit
model = LogisticRegression()
model.fit(X=X_train, y=y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred

# Conclusions and metrics
# Check the accuracy score:
accuracy_score(y_test, y_pred) 

print(f'accuracy_score: {accuracy_score}. As the dataset is imbalanced, probably the accuracy is not the best metric')
'''
True Negatives = 1591
False Negatives = 191
False Positives = 209
True Positives = 967
'''
print(f'There are {1591+967} correct predictions')
print(f'There are {209+191} incorrect predictions')

# concatenate y_test and y_pred:
df_pred = pd.DataFrame({'Real': np.array(y_test), 'Predicted': np.array(y_pred)})
print(classification_report(df_pred['Real'], df_pred['Predicted']))

print('Creo que la prediccion es bastante buena. De todos modos voy a aplicar un Grid Search para mejorar.')


# Grid Search
solvers = ['newton-cg', 'lbfgs', 'liblinear'] # Most 3 common solvers for LogisticRegression
penalty = ['l2'] # regularización
c_values = [100, 10, 1.0, 0.1, 0.01]

grid = dict(solver=solvers, penalty=penalty, C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=13) 
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X, y)

# results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("best: # Best: 0.861771 using {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}")


# optimized model:
optimized_model = LogisticRegression(C=10, penalty='l2', solver='lbfgs')
optimized_model.fit(X_train, y_train)
y_pred2 = optimized_model.predict(X_test)

# accuracy
accuracy_score(y_pred2, y_test)
'''
True Negatives = 1583
False Negatives = 201
False Positives = 217
True Positives = 957
'''
print(f'There are {1583+957} correct predictions')
print(f'There are {201+217} incorrect predictions')

print(' EL MODELO EMPEORÓ.')

# concatenate y_test and y_pred
df_pred2 = pd.DataFrame({'Real_2': np.array(y_test), 'Predicted_2': np.array(y_pred2)})
print(classification_report(df_pred2['Real_2'], df_pred2['Predicted_2']))