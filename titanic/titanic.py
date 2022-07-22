import numpy as np
import pandas as pd
import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer

def main():
  train = pd.read_csv("./data/train.csv")
  # print(train.head())

  test = pd.read_csv("./data/test.csv")
  # print(test.head())

  # gender_submission = pd.read_csv("./data/gender_submission.csv")
  # print(gender_submission.head())


  # 欠損値割合の確認
  chk_null = train.isnull().sum()
  chk_null_pct = chk_null / (train.index.max() + 1)
  chk_null_tbl = pd.concat([chk_null[chk_null > 0], chk_null_pct[chk_null_pct > 0] * 100], axis=1)
  chk_null_tbl = chk_null_tbl.rename(columns={0: "欠損レコード数",1: "欠損割合(missing rows / all rows)"})
  print(chk_null_tbl)

  train = pd.get_dummies(train, columns=['Sex', 'Embarked'])
  for t in train:
    train[t] = train[t].fillna(train[t].mode()[0])

  test = pd.get_dummies(test, columns=['Sex', 'Embarked'])
  for t in test:
    test[t] = test[t].fillna(test[t].mode()[0])

  X_train = train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]]
  y_train = train["Survived"]

  X_test = test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]]

  clf = KNeighborsClassifier(n_neighbors=1)
  clf.fit(X_train, y_train)

  # t_test に予測結果を格納
  y_test = clf.predict(X_test)

  # PassengerId を取得
  PassengerId = np.array(test["PassengerId"]).astype(int)

  # y_test と PassengerId を結合
  answer = pd.DataFrame(y_test, PassengerId, columns = ["Survived"])

  now = datetime.datetime.now()
  today = now.strftime('%Y-%m-%d_%H-%M-%S')
  # answer.to_csv("titanic_answer_" + today + ".csv", index_label = ["PassengerId"])


  corr = X_train.corr()
  # print(corr)
  # corr_minus = (corr < -0.5)
  # corr_plus = (corr > 0.5) & (corr != 1)
  print(corr.where(corr < -0.5, np.NaN))
  # print(corr_plus.sum())


if __name__ == '__main__':
  main()
