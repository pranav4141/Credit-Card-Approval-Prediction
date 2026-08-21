import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    train = pd.read_csv(
        "https://raw.githubusercontent.com/pranav4141/Credit-Card-Approval-Prediction/refs/heads/main/datasets/train.csv"
    )
    test = pd.read_csv(
        "https://raw.githubusercontent.com/pranav4141/Credit-Card-Approval-Prediction/refs/heads/main/datasets/test.csv"
    )
    return train, test


train_original, test_original = load_data()

full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()
test_copy = test_original.copy()


def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    result = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    result.columns = ["Count", "Frequency (%)"]
    return result


# -----------------------------
# PREPROCESSING
# -----------------------------
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers=None):
        self.feat_with_outliers = feat_with_outliers or [
            "Family member count", "Income", "Employment length"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if set(self.feat_with_outliers).issubset(X.columns):
            Q1 = X[self.feat_with_outliers].quantile(0.25)
            Q3 = X[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            X = X[
                ~(
                    (X[self.feat_with_outliers] < (Q1 - 3 * IQR))
                    | (X[self.feat_with_outliers] > (Q3 + 3 * IQR))
                ).any(axis=1)
            ]
        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        self.feature_to_drop = feature_to_drop or [
            "Has a mobile phone",
            "Children count",
            "Job title",
            "Account age",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.feature_to_drop if c in X.columns]
        if cols:
            X.drop(cols, axis=1, inplace=True)
        return X


class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days=None):
        self.feat_with_days = feat_with_days or [
            "Employment length", "Age"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.feat_with_days if c in X.columns]
        if cols:
            X[cols] = np.abs(X[cols])
        return X


class RetireeHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Employment length" in X.columns:
            X.loc[
                X["Employment length"] == 365243,
                "Employment length"
            ] = 0
        return X


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_skewness=None):
        self.feat_with_skewness = feat_with_skewness or ["Income", "Age"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.feat_with_skewness if c in X.columns]
        if cols:
            X[cols] = np.cbrt(X[cols])
        return X


class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc=None):
        self.feat_with_num_enc = feat_with_num_enc or [
            "Has a work phone", "Has a phone", "Has an email"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for ft in self.feat_with_num_enc:
            if ft in X.columns:
                X[ft] = X[ft].map({1: "Y", 0: "N"})
        return X


class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=None):
        self.one_hot_enc_ft = one_hot_enc_ft or [
            "Gender", "Marital status", "Dwelling", "Employment status",
            "Has a car", "Has a property", "Has a work phone",
            "Has a phone", "Has an email"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        cols = [c for c in self.one_hot_enc_ft if c in X.columns]
        if cols:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(X[cols])
            names = encoder.get_feature_names_out(cols)
            encoded_df = pd.DataFrame(
                encoded, columns=names, index=X.index
            )
            X = pd.concat(
                [encoded_df, X.drop(columns=cols)],
                axis=1
            )
        return X


class OrdinalFeatNames(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Education level" in X.columns:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
            X[["Education level"]] = encoder.fit_transform(
                X[["Education level"]]
            )
        return X


class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=None):
        self.min_max_scaler_ft = min_max_scaler_ft or [
            "Age", "Income", "Employment length"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.min_max_scaler_ft if c in X.columns]
        if cols:
            scaler = MinMaxScaler()
            X[cols] = scaler.fit_transform(X[cols])
        return X


class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Is high risk" in X.columns:
            X["Is high risk"] = pd.to_numeric(X["Is high risk"])
        return X


class OversampleSMOTE(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if "Is high risk" not in X.columns:
            return X

        X = X.copy()
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(
            X.drop(columns=["Is high risk"]),
            X["Is high risk"]
        )
        result = pd.DataFrame(X_bal)
        result["Is high risk"] = y_bal
        return result


def full_pipeline(df):
    pipeline = Pipeline([
        ("outlier_remover", OutlierRemover()),
        ("feature_dropper", DropFeatures()),
        ("time_conversion_handler", TimeConversionHandler()),
        ("retiree_handler", RetireeHandler()),
        ("skewness_handler", SkewnessHandler()),
        ("binning_num_to_yn", BinningNumToYN()),
        ("one_hot_with_feat_names", OneHotWithFeatNames()),
        ("ordinal_feat_names", OrdinalFeatNames()),
        ("min_max_with_feat_names", MinMaxWithFeatNames()),
        ("change_to_num_target", ChangeToNumTarget()),
        ("oversample_smote", OversampleSMOTE()),
    ])
    return pipeline.fit_transform(df)


# -----------------------------
# TRAIN MODEL ONCE
# -----------------------------
@st.cache_resource
def train_model(train_df):
    prepared = full_pipeline(train_df.copy())

    X = prepared.drop(columns=["Is high risk"])
    y = prepared["Is high risk"]

    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    model.fit(X, y)

    return model, X.columns.tolist()


model, model_columns = train_model(train_copy)


# -----------------------------
# LOTTIE
# -----------------------------
@st.cache_data
def load_lottieurl(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


lottie_loading_an = load_lottieurl(
    "https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json"
)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Credit Card Approval Prediction")
st.write(
    "This app predicts if an applicant will be approved for a credit card "
    "based on the information provided below."
)

st.subheader("Applicant Information")

input_gender = st.radio(
    "Select your gender",
    ["Male", "Female"],
    index=0
)

age_years = st.slider(
    "Select your age",
    value=42,
    min_value=18,
    max_value=70,
    step=1
)
input_age = -age_years * 365.25

marital_status_key = [
    "Married",
    "Single/not married",
    "Civil marriage",
    "Separated",
    "Widowed",
]
marital_status_values = list(
    value_cnt_norm_cal(full_data, "Marital status").index
)
marital_status_dict = dict(
    zip(marital_status_key, marital_status_values)
)
input_marital_status_key = st.selectbox(
    "Select your marital status",
    marital_status_key
)
input_marital_status_val = marital_status_dict[
    input_marital_status_key
]

fam_member_count = float(
    st.selectbox(
        "Select your family member count",
        [1, 2, 3, 4, 5, 6]
    )
)

dwelling_type_key = [
    "House / apartment",
    "Live with parents",
    "Municipal apartment ",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
dwelling_type_values = list(
    value_cnt_norm_cal(full_data, "Dwelling").index
)
dwelling_type_dict = dict(
    zip(dwelling_type_key, dwelling_type_values)
)
input_dwelling_type_key = st.selectbox(
    "Select the type of dwelling you reside in",
    dwelling_type_key
)
input_dwelling_type_val = dwelling_type_dict[
    input_dwelling_type_key
]

input_income = st.number_input(
    "Enter your income (in USD)",
    min_value=0,
    value=0,
    step=100
)

employment_status_key = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Student",
]
employment_status_values = list(
    value_cnt_norm_cal(full_data, "Employment status").index
)
employment_status_dict = dict(
    zip(employment_status_key, employment_status_values)
)
input_employment_status_key = st.selectbox(
    "Select your employment status",
    employment_status_key
)
input_employment_status_val = employment_status_dict[
    input_employment_status_key
]

employment_years = st.slider(
    "Select your employment length",
    value=6,
    min_value=0,
    max_value=30,
    step=1
)
input_employment_length = -employment_years * 365.25

edu_level_key = [
    "Secondary school",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
edu_level_values = list(
    value_cnt_norm_cal(full_data, "Education level").index
)
edu_level_dict = dict(
    zip(edu_level_key, edu_level_values)
)
input_edu_level_key = st.selectbox(
    "Select your education status",
    edu_level_key
)
input_edu_level_val = edu_level_dict[
    input_edu_level_key
]

input_car_ownship = st.radio(
    "Do you own a car?",
    ["Yes", "No"],
    index=0
)

input_prop_ownship = st.radio(
    "Do you own a property?",
    ["Yes", "No"],
    index=0
)

input_work_phone = st.radio(
    "Do you have a work phone?",
    ["Yes", "No"],
    index=0
)
work_phone_val = {"Yes": 1, "No": 0}[input_work_phone]

input_phone = st.radio(
    "Do you have a phone?",
    ["Yes", "No"],
    index=0
)
phone_val = {"Yes": 1, "No": 0}[input_phone]

input_email = st.radio(
    "Do you have an email?",
    ["Yes", "No"],
    index=0
)
email_val = {"Yes": 1, "No": 0}[input_email]


# -----------------------------
# PREDICTION
# -----------------------------
predict_bt = st.button(
    "Predict",
    type="primary",
    use_container_width=True
)


def prepare_profile():
    profile = [
        0,
        input_gender[:1],
        input_car_ownship[:1],
        input_prop_ownship[:1],
        0,
        input_income,
        input_employment_status_val,
        input_edu_level_val,
        input_marital_status_val,
        input_dwelling_type_val,
        input_age,
        input_employment_length,
        1,
        work_phone_val,
        phone_val,
        email_val,
        "to_be_droped",
        fam_member_count,
        0.00,
        0,
    ]

    profile_df = pd.DataFrame(
        [profile],
        columns=train_copy.columns
    )

    return profile_df


def prepare_profile_for_model(profile_df):
    # Apply the same feature transformations used during training.
    df = profile_df.copy()

    df = OutlierRemover().transform(df)
    df = DropFeatures().transform(df)
    df = TimeConversionHandler().transform(df)
    df = RetireeHandler().transform(df)
    df = SkewnessHandler().transform(df)
    df = BinningNumToYN().transform(df)

    # Encoding/scaling must use the training data categories/ranges.
    # Re-run the original preprocessing together with the profile so
    # that the resulting columns match the trained model.
    combined = pd.concat(
        [train_copy.copy(), profile_df],
        ignore_index=True
    )

    prepared = full_pipeline(combined)

    profile_prepared = prepared.iloc[[-1]].copy()

    return profile_prepared.drop(columns=["Is high risk"])


if predict_bt:
    with st.spinner("Making prediction..."):
    try:
        profile_df = prepare_profile()
        profile_prepared = prepare_profile_for_model(profile_df)

        profile_prepared = profile_prepared.reindex(
            columns=model_columns,
            fill_value=0
        )

        final_pred = model.predict(profile_prepared)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        final_pred = None

    if final_pred is not None:
        if final_pred[0] == 0:
            st.success("## You have been approved for a credit card")
            st.balloons()
        else:
            st.error(
                "## Unfortunately, you have not been approved for a credit card"
            )
