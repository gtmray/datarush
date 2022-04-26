import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import sys

# Reading the original data
data_path_main = str(sys.argv[1])
main_train_df = pd.read_csv(data_path_main)

# Reading the manipulated data
data_path_train =  str(sys.argv[2])
train_df = pd.read_csv(data_path_train)


# Reading the testing data
data_path_test = str(sys.argv[3])
solution_df = pd.read_csv(data_path_test)

# Drop NaN values if any

train_df.dropna(inplace=True)

# Classes with data less than 50 in orignal data are dropped in manipulated data

small_classes = main_train_df.category_num.value_counts() < 50
drop_classes = [i for i in small_classes.index if small_classes[i]]
index_to_drop = train_df.loc[train_df.category_num.isin(drop_classes)].index
train_df.drop(index_to_drop, inplace=True)
train_df.category_num.value_counts()

print("DATA HAS BEEN READ")
tfidf = TfidfVectorizer(max_features=200000, sublinear_tf=True, 
                        stop_words='english', dtype='float32')
tfidf.fit(train_df['abstract'])
X_train_tfidf = tfidf.transform(train_df['abstract'])
solution_tfidf = tfidf.transform(solution_df['abstract'])

X_train = X_train_tfidf
y_train = train_df['category_num']
model = LinearSVC(random_state=77)
model.fit(X_train, y_train)

y_pred = model.predict(solution_tfidf)
solution_df['category_num'] = y_pred
solution_df = solution_df[['id', 'category_num']]
solution_df.to_csv('solution.csv', index=False)
print("solution.csv saved")
