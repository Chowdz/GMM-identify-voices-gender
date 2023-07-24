"""
# encoding: utf-8
#!/usr/bin/env python3

@Author : ZDZ
@Time : 2022/10/12 18:06 
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import time

voice = pd.DataFrame(pd.read_csv('voice.csv'))
title = voice.columns.values
X = voice.iloc[:, :-1]
Y = voice.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

plt.rcParams['figure.figsize'] = (15, 10)
sns.heatmap(round(abs(voice.corr()), 3), annot=True, linewidths=.5, cmap="GnBu")
plt.title('Correlation between features', fontsize=20)
plt.show()

time_begin1 = time.time()
model1 = XGBClassifier()
model1.fit(X_train, Y_train)
y_test_predict = model1.predict(X_test)
acc = metrics.accuracy_score(y_test_predict, Y_test)
plot_importance(model1)
plt.show()
time_end1 = time.time()
time1 = time_end1 - time_begin1
print('XG_boost耗时：', end="")
print(f"\033[1;31m{round(time1, 4)}\033[0m", end="")
print('秒')
print('准确率为：', end="")
print(f"\033[1;31m{acc}\033[0m")
print("分类报告： ", metrics.classification_report(Y_test, y_test_predict))

time_begin2 = time.time()
model2 = LogisticRegression(solver='newton-cg', multi_class='multinomial', random_state=1)
model2.fit(X_train, Y_train)
y_test_predict = model2.predict(X_test)
acc = metrics.accuracy_score(y_test_predict, Y_test)
time_end2 = time.time()
time2 = time_end2 - time_begin2
print('逻辑回归耗时：', end="")
print(f"\033[1;31m{round(time2, 4)}\033[0m", end="")
print('秒')
print('准确率为：', end="")
print(f"\033[1;31m{acc}\033[0m")
print("分类报告： ", metrics.classification_report(Y_test, y_test_predict))

time_begin4 = time.time()
model4 = tree.DecisionTreeClassifier(random_state=1)
model4.fit(X_train, Y_train)
y_test_predict = model4.predict(X_test)
acc = metrics.accuracy_score(y_test_predict, Y_test)
time_end4 = time.time()
time4 = time_end4 - time_begin4
print('决策树耗时：', end="")
print(f"\033[1;31m{round(time4, 4)}\033[0m", end="")
print('秒')
print('准确率为：', end="")
print(f"\033[1;31m{acc}\033[0m")
print("分类报告： ", metrics.classification_report(Y_test, y_test_predict))

time_begin5 = time.time()
model5 = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=1)
model5.fit(X_train, Y_train)
y_test_predict = model5.predict(X_test)
acc = metrics.accuracy_score(y_test_predict, Y_test)
time_end5 = time.time()
time5 = time_end5 - time_begin5
print('随机森林耗时：', end="")
print(f"\033[1;31m{round(time5, 4)}\033[0m", end="")
print('秒')
print('准确率为：', end="")
print(f"\033[1;31m{acc}\033[0m")
print("分类报告： ", metrics.classification_report(Y_test, y_test_predict))

time_begin6 = time.time()
model6 = SVC(kernel='rbf', gamma=0.01, C=10, random_state=1)
model6.fit(X_train, Y_train)
y_test_predict = model6.predict(X_test)
acc = metrics.accuracy_score(y_test_predict, Y_test)
time_end6 = time.time()
time6 = time_end6 - time_begin6
print('支持向量机耗时：', end="")
print(f"\033[1;31m{round(time6, 4)}\033[0m", end="")
print('秒')
print('准确率为：', end="")
print(f"\033[1;31m{acc}\033[0m")
print("分类报告： ", metrics.classification_report(Y_test, y_test_predict))
