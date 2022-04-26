import math
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
import sys


# Main dataset


train_path = str(sys.argv[1])
train_df = pd.read_csv(train_path)

train_df = train_df[['abstract', 'category_num']]

print("DATA HAS BEEN READ")
print("STARTING AUGMENTATION WHICH TAKES SEVERAL MINUTES")
# Function for augmentation

def augmentation_text(aug, df: pd.DataFrame, factor: pd.Series, labels_to_aug: list):
    augmented_sentences=[]
    augmented_sentences_labels=[]

    for idx, row in df.iterrows():
        if row[1] in labels_to_aug:
            factor_to_aug = math.ceil(factor[row[1]])
            temps = aug.augment(row[0], n=factor_to_aug)
            for sentences in temps[1:]:
                if factor_to_aug>1:
                    augmented_sentences.append(sentences)
                    augmented_sentences_labels.append(row[1])

    return augmented_sentences, augmented_sentences_labels

# Data augmentation using synonyms

lt_600 = (train_df.category_num.value_counts(ascending=True) < 550)
lt_600_labels = [i for i in lt_600.index if lt_600[i]]

factor = (550/train_df.category_num.value_counts(ascending=True))

max_word_aug = 30
aug = naw.SynonymAug(aug_src='wordnet',aug_max=max_word_aug)

sent, labels = augmentation_text(aug, train_df, factor, lt_600_labels)
df_aug = pd.DataFrame({'abstract': sent,
                       'category_num': labels})

train_df = pd.concat([train_df, df_aug])
print("1/3 augmentation done")

# Data augmentation by swapping words

max_word_aug = 30
aug_swap = naw.random.RandomWordAug(action='swap',aug_max=max_word_aug)

factor_swap = (1000/train_df.category_num.value_counts(ascending=True))

sent, labels = augmentation_text(aug_swap, train_df, factor_swap, lt_600_labels)
df_aug_swap = pd.DataFrame({'abstract': sent,
                       'category_num': labels})

train_df = pd.concat([train_df, df_aug_swap])
print("2/3 augmentation done")

# Data augmentation using synonyms to categories whose count is less than 1000

lt_1300 = (train_df.category_num.value_counts(ascending=True) < 1000)
lt_1300_labels = [i for i in lt_1300.index if lt_1300[i]]

factor = (1000/train_df.category_num.value_counts(ascending=True))
max_word_aug = 30
aug = naw.SynonymAug(aug_src='wordnet',aug_max=max_word_aug)

sent, labels = augmentation_text(aug, train_df, factor, lt_1300_labels)
df_aug_1300 = pd.DataFrame({'abstract': sent,
                       'category_num': labels})
print("ALL AUGMENTATION DONE")

train_df = pd.concat([train_df, df_aug_1300])

# Saving to csv after 3 augmentation 

train_df.to_csv('data/updated/augmented.csv', index=False)

print("AUGMENTED FILE SAVED")
