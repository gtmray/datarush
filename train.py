import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
import sys

# Reading the original data
data_path_main = str(sys.argv[1])
main_train_df = pd.read_csv(data_path_main)

# Reading the manipulated data
data_path_train =  str(sys.argv[2])
data_path_test = str(sys.argv[3])

train_df = pd.read_csv(data_path_train)
test_df = pd.read_csv(data_path_test)

# Drop NaN values if any

train_df.dropna(inplace=True)

# Classes with data less than 50 in orignal data are dropped in manipulated data

small_classes = main_train_df.category_num.value_counts() < 50
drop_classes = [i for i in small_classes.index if small_classes[i]]
index_to_drop = train_df.loc[train_df.category_num.isin(drop_classes)].index
train_df.drop(index_to_drop, inplace=True)

print("DATA HAS BEEN READ. STARTING VECTORIZATION")
tfidf = TfidfVectorizer(max_features=200000, sublinear_tf=True, 
                        stop_words='english', dtype='float32')
tfidf.fit(train_df['abstract'])
X_train_tfidf = tfidf.transform(train_df['abstract'])
X_test_tfidf = tfidf.transform(test_df['abstract'])

X_train = X_train_tfidf
y_train = train_df['category_num']
X_test = X_test_tfidf
y_test = test_df['category_num']

print("STARTING TO TRAIN MODEL")
model = LinearSVC(random_state=77)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
print("TRAINED!")
print("Scores on validation data: ")
print(f"Accuracy score: {acc}")
score = f1_score(y_pred, y_test, average='weighted')
print(f"F1 Weighted Score: {score}")
score_macro = f1_score(y_pred, y_test, average='macro')
print(f"F1 Macro Score: {score_macro}")

pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print("MODEL SAVED")