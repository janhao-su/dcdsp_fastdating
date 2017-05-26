import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    f = open('output.txt', 'w')
    print(x)
    print >> f, x
    pd.reset_option('display.max_rows')
    f.close()

data_df = pd.read_csv("speed_dating_train.csv", encoding="ISO-8859-1")
# take a quick look into Data
print(data_df.head())	
print(data_df.describe())

# see NaN
print_full(data_df.isnull().sum())

# delete unwanted columns and wave 6 to 9
nonvars=["iid", "id", "idg", "condtn", "position", "positin1", "partner", "pid", "dec_o", "like_o", "prob_o", "met_o", "field", "undergra", "mn_sat", "tuition", "from", "income", "goal", "date", "go_out", "career", "sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga", "exphappy", "expnum", "zipcode", "attr4_1", "sinc4_1", "intel4_1", "fun4_1", "amb4_1", "shar4_1", "attr5_1", "sinc5_1", "intel5_1", "fun5_1", "amb5_1", "dec", "like", "prob", "met", "match_es", "attr1_s", "sinc1_s", "intel1_s", "fun1_s", "amb1_s", "shar1_s", "attr3_s", "sinc3_s", "intel3_s", "fun3_s", "amb3_s", "satis_2", "length", "numdat_2", "attr7_2", "sinc7_2", "intel7_2", "fun7_2", "amb7_2", "shar7_2", "attr1_2", "sinc1_2", "intel1_2", "fun1_2", "amb1_2", "shar1_2", "attr4_2", "sinc4_2", "intel4_2", "fun4_2", "amb4_2", "shar4_2", "attr2_2", "sinc2_2", "intel2_2", "fun2_2", "amb2_2", "shar2_2", "attr3_2", "sinc3_2", "intel3_2", "fun3_2", "amb3_2", "attr5_2", "sinc5_2", "intel5_2", "fun5_2", "amb5_2", "you_call", "them_cal", "date_3", "numdat_3", "num_in_3", "attr1_3", "sinc1_3", "intel1_3", "fun1_3", "amb1_3", "shar1_3", "attr7_3", "sinc7_3", "intel7_3", "fun7_3", "amb7_3", "shar7_3", "attr4_3", "sinc4_3", "intel4_3", "fun4_3", "amb4_3", "shar4_3", "attr2_3", "sinc2_3", "intel2_3", "fun2_3", "amb2_3", "shar2_3", "attr3_3", "sinc3_3", "intel3_3", "fun3_3", "amb3_3", "attr5_3", "sinc5_3", "intel5_3", "fun5_3", "amb5_3"]
df_exclude=data_df.drop(data_df[(data_df.wave>5) & (data_df.wave<10)].index).drop(nonvars, 1)
# delete rows contain NaN
nan_row_index = pd.isnull(df_exclude).any(1).nonzero()[0]
df_clean = df_exclude.drop(df_exclude.index[nan_row_index])

# divide data into train set and test set (wave 21)
exclude_wave21_df = df_clean.drop(df_clean[df_clean.wave==21].index)
match_df = exclude_wave21_df.drop(exclude_wave21_df[exclude_wave21_df.match==0].index)
not_match_df = exclude_wave21_df.drop(exclude_wave21_df[exclude_wave21_df.match==1].index)
train_df = not_match_df.sample(710).append(match_df).drop("wave", 1)

test_df = df_clean.drop(df_clean[(df_clean.wave<21)].index).drop("wave", 1)

from sklearn.svm import SVC
train_df_X = train_df.drop("match", 1)
train_df_y = train_df.match
test_df_X = test_df.drop("match", 1)
test_df_y = test_df.match
clf = SVC(kernel='poly', degree=1)
print(clf.fit(train_df_X, train_df_y).score(test_df_X, test_df_y)) 


