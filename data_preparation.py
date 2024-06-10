import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
import pickle

RANDOM_STATE = 42


def write_to_pickle(filename, nd_array):
    # write to pickle
    with open(filename, 'wb') as f:
        pickle.dump(nd_array, f, pickle.HIGHEST_PROTOCOL)


def main():
    # edit pandas display options to show all cols
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 1500)

    # load raw data from csv file # healthcare-dataset-stroke-data.csv, manually_created-stroke-data.csv
    stroke_data_df = pd.read_csv('datasets/healthcare-dataset-stroke-data.csv',
                                 sep=',')

    # PREPROCESSING
    # delete id column, since this column is not relevant for the model
    stroke_data_df.drop('id', axis=1, inplace=True)

    # NOTE: dropping gender, hypertension, and heart_disease does not increase the AVG Recall and Precision Score
    # drop gender
    # stroke_data_df.drop('gender', axis=1, inplace=True)

    # drop hypertension
    # stroke_data_df.drop('hypertension', axis=1, inplace=True)

    # drop heart_disease
    # stroke_data_df.drop('heart_disease', axis=1, inplace=True)

    # NOTE: dropping ever_married,Residence_type and work_type slightly increases the AVG Recall and Precision Score
    # drop ever_married
    # stroke_data_df.drop('ever_married', axis=1, inplace=True)

    # drop Residence_type
    # stroke_data_df.drop('Residence_type', axis=1, inplace=True)

    # drop work_type
    # stroke_data_df.drop('work_type', axis=1, inplace=True)

    # drop smoking_status
    # stroke_data_df.drop('smoking_status', axis=1, inplace=True)

    # NOTE:
    # delete entry with gender "Other"
    # stroke_data_df.drop(stroke_data_df[stroke_data_df['gender'] == 'Other'].index, axis=0,
    #                    inplace=True)  # new test samples with gender "Other" should get classified though

    # general info on features (datatype, missing values etc)
    stroke_data_df.info()

    # cast age column from float to int, age from 0 to 2 is in fractions, floor these by casting to int
    stroke_data_df["age"] = stroke_data_df["age"].astype('int16')

    # filter only age > X
    # NOTE: does not decrease the number of False Negatives
    # stroke_data_df = stroke_data_df[stroke_data_df["age"] > 50]

    is_nan_rows = stroke_data_df[stroke_data_df["bmi"].isna()]
    # show rows where BMI has missing values
    print("BMI ISNAN:\n", is_nan_rows.count())
    print("BMI IS NAN rows:\n", is_nan_rows)

    # outlier overview for continuous data (min,max,median etc)
    print(stroke_data_df.describe())

    # split into X and y, last column is target column
    y = stroke_data_df.pop("stroke")
    X = stroke_data_df

    # split Train and Test Data stratified by "stroke" (target) column
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=0.20, random_state=RANDOM_STATE)

    print("X-TRAIN BEFORE ENCODING:\n", X_train)

    # transform some categorical values into numerical with OrdinalEncoder
    ordinal_encode_columns = ["smoking_status", "ever_married", "Residence_type"]  # "gender"
    ordinal_categories = [
        # ["Female", "Male", "Other"],
        ["never smoked", "Unknown", "formerly smoked", "smokes"],
        ["No", "Yes"],
        ["Urban", "Rural"]
    ]

    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories)
    ordinal_encoder.fit(X_train[ordinal_encode_columns])
    X_train[ordinal_encode_columns] = ordinal_encoder.transform(X_train[ordinal_encode_columns])
    X_test[ordinal_encode_columns] = ordinal_encoder.transform(X_test[ordinal_encode_columns])
    print(f"Features '{'\', \''.join(ordinal_encode_columns)}' ORDINAL ENCODED:\n", X_train)

    # transform work_type categorical values into numerical with OneHotEncoder
    onehot_encode_multivariate_columns = ["work_type", "gender"]
    one_hot_encoder = OneHotEncoder(handle_unknown='error', sparse_output=False)
    one_hot_encoder.fit(X_train[onehot_encode_multivariate_columns])
    categories_raveled = np.concatenate(one_hot_encoder.categories_).ravel()
    X_train_gender_ohe_matrix = one_hot_encoder.transform(X_train[onehot_encode_multivariate_columns])
    print("ONEHOT ENCODED NP MATRIX:\n", X_train_gender_ohe_matrix)
    X_train_gender_ohe_df = pd.DataFrame(index=X_train.index, data=X_train_gender_ohe_matrix,
                                         columns=categories_raveled)
    X_train = X_train.join(X_train_gender_ohe_df)
    X_train.drop(onehot_encode_multivariate_columns, axis=1, inplace=True)

    X_test_gender_ohe_matrix = one_hot_encoder.transform(X_test[onehot_encode_multivariate_columns])
    X_test_gender_ohe_df = pd.DataFrame(index=X_test.index, data=X_test_gender_ohe_matrix,
                                        columns=categories_raveled)
    X_test = X_test.join(X_test_gender_ohe_df)
    X_test.drop(onehot_encode_multivariate_columns, axis=1, inplace=True)

    print(f"Features '{'\', \''.join(onehot_encode_multivariate_columns)}' ONEHOT ENCODED:\n", X_train)
    X_train_named_columns = X_train.columns
    print("Feature names:", X_train_named_columns)
    # impute missing values with IterativeImputer ("bmi" column)
    impute_columns = ["bmi"]
    imputer = IterativeImputer(missing_values=np.nan, random_state=RANDOM_STATE)
    # imputer = SimpleImputer(strategy="most_frequent")
    # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform", metric="nan_euclidean")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    # resample data with undersampling
    # NOTE: 1:1 NearMiss undersampling is able to reduce False Negatives significantly, but also increases False positives (better recall but worse precision)
    # resampler = NearMiss(sampling_strategy=1.0, version=2, n_neighbors=3)
    # resampler = ClusterCentroids(estimator=MiniBatchKMeans(n_init=3, random_state=RANDOM_STATE),
    #                             random_state=RANDOM_STATE)
    # resampler = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    # X_train, y_train = resampler.fit_resample(X_train, y_train)

    # resample data with oversampling

    write_to_pickle("pickledata/X_train.pickle", X_train)
    write_to_pickle("pickledata/y_train.pickle", y_train)
    write_to_pickle("pickledata/X_test.pickle", X_test)
    write_to_pickle("pickledata/y_test.pickle", y_test)
    print("Picklefiles written!")
    print("Data preparation finished!")


if __name__ == "__main__":
    main()
