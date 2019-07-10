# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:38:56 2019

@author: siva1
"""

import pandas as pd
from six.moves import urllib
import shutil
import tensorflow as tf


Train_File_Name = "census/adult.data"
Test_File_Name = "census/adult.test"

urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", Train_File_Name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", Test_File_Name)


CSV_Columns = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

df = pd.read_csv(Train_File_Name, names = CSV_Columns, skipinitialspace = True, skiprows = 1)

Trimmed_Columns = ["age", "workclass", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "hours_per_week", "native_country", "income_bracket"]

df = df[Trimmed_Columns]

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ['Male', 'Female'])
race = tf.feature_column.categorical_column_with_vocabulary_list("race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
education = tf.feature_column.categorical_column_with_vocabulary_list("education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list("marital_status", ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])
relationship = tf.feature_column.categorical_column_with_vocabulary_list("relationship", ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
workclass = tf.feature_column.categorical_column_with_vocabulary_list("workclass", ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

age_buckets = tf.feature_column.bucketized_column(age, boundaries= [18,25, 30, 35, 40, 45, 50, 55, 60, 65])

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

base_columns = [gender, race, marital_status, workclass, occupation, native_country, age_buckets, education]

crossed_columns = [
        tf.feature_column.crossed_column(["education", "occupation"], hash_bucket_size=1000),
        tf.feature_column.crossed_column([age_buckets, "education", "occupation"], hash_bucket_size=1000),
        tf.feature_column.crossed_column(["native_country", "occupation"], hash_bucket_size= 1000)
        ]

deep_columns = [education_num, hours_per_week]

def input_fn(File_Name, num_epochs, shuffle):
    df = pd.read_csv(File_Name, names = CSV_Columns, skipinitialspace=True, skiprows=1)
    df = df[Trimmed_Columns]
    
    df = df.dropna(how = 'any', axis = 0)
    
    labels = df["income_bracket"].apply(lambda row: ">50k" in row).astype(int)
    
    return tf.estimator.inputs.pandas_input_fn(
            x = df,
            y = labels,
            batch_size = 100, 
            num_epochs = num_epochs,
            shuffle = shuffle,
            num_threads = 5)
    
Model_DIR = "./Linear_classifier"
shutil.rmtree(Model_DIR,ignore_errors = True)

Linear_Model = tf.estimator.LinearClassifier(model_dir= Model_DIR, feature_columns= base_columns + crossed_columns +deep_columns)

train = Linear_Model.train(input_fn(Train_File_Name, num_epochs=None, shuffle=True), steps=1000)

results = Linear_Model.evaluate(input_fn(Test_File_Name, num_epochs=1, shuffle=False), steps=None)

for key in sorted(results):
    print(key, results[key])
    


        



