from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
)
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.utils import estimator_html_repr
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

from utils.datainformation import DataInformation
from utils.pipelineinformation import PipelineInformation


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


def _init_header():
    SUPTITLE = 'Supervised Learning Sand-Box'
    st.title(SUPTITLE)

def _init_sidebar() -> DataInformation:
    SIDEBAR_SUPTITLE = 'Choose Data Science problem'
    st.sidebar.title(SIDEBAR_SUPTITLE)

    # available datasets
    OPENML_DATASET_NAMES = [
        'titanic',
        'credit-g',
        'liver-disorders',
        'cholesterol',
        'spambase',
        'bodyfat'
    ]

    # choose dataset
    openml_dataset_name = st.sidebar.selectbox('Select A Toy Data Set', OPENML_DATASET_NAMES)
    uploaded_dataset_name = st.sidebar.file_uploader('Or upload your own')

    df, dataset_name = (
        (pd.read_csv(uploaded_dataset_name), uploaded_dataset_name)
        if uploaded_dataset_name else
        (sklearn_to_df(fetch_openml(openml_dataset_name, version=1, as_frame=True, return_X_y=False)),
         openml_dataset_name)
    )

    columns = df.columns.to_list()
    # choose features and target column
    default_index = columns.index('target') if "target" in columns else 0
    target: str = st.sidebar.selectbox('Choose Target Name', columns, default_index)

    features_all = columns.copy()
    features_all.remove(target)
    features: List[str] = st.sidebar.multiselect('Choose Features', options=features_all, default=features_all)

    # model customization
    st.sidebar.header('Model settings')

    MODELS = [
        'Linear Regression',
        'Ridge'
    ]
    estimator_name = st.sidebar.selectbox('Choose Estimator', MODELS)

    # TODO: add more options
    _ = st.sidebar.selectbox('ADD MORE OPTIONS HERE', ["More settings soon..."])

    return DataInformation(df, features, target, dataset_name)


def get_model(classification: bool):
    return LogisticRegression() if classification else LinearRegression()


def determine_task(data_info: DataInformation):
    max_unique_vals = 15 # heuristic
    return len(data_info.df[data_info.target].unique()) <= max_unique_vals


def _build_pipeline(data_info: DataInformation):
    set_config(display="diagram")

    df = data_info.df

    #
    numerical_features = (
        df[data_info.features]
        .select_dtypes('number')
        .columns
        .to_list()
    )
    categorical_features = (
        df[data_info.features]
        .columns
        .difference(numerical_features)
        .to_list()
    )

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy = 'mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numerical_features),
            ('categorical', categorical_transformer, categorical_features)
        ]
    )

    # determine whether it is a regression or classification problem
    classification = determine_task(data_info=data_info)

    # create estimator
    estimator = get_model(classification)

    # create pipeline
    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('estimator', estimator)
        ]
    )

    return PipelineInformation(pipeline, classification)


def _plot_results(
        pipeline_info: PipelineInformation,
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_hat: pd.DataFrame,
        num_folds: int
    ):
    st.header('Results')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    left, right = st.columns(2)

    pipeline = pipeline_info.pipeline
    with right:
        if pipeline_info.classification and len(y_test.unique()) <= 2:
            st.header('ROC Curve')
            fig_1 = (
                RocCurveDisplay.
                from_estimator(pipeline, X_test, y_test)
                .plot()
            )
            st.pyplot()

    with left:
        if pipeline_info.classification:
            st.header('Confusion Matrix')
            cm = confusion_matrix(y_test, y_hat, labels=pipeline.named_steps['estimator'].classes_)
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.named_steps['estimator'].classes_).plot()
            st.pyplot()

    # scores
    scoring = (
        ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
        if pipeline_info.classification else
        ['r2', 'explained_variance', 'neg_mean_absolute_error']
   )

    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=num_folds)
    score_df = pd.DataFrame(scores)
    score_df = pd.concat([score_df, score_df.agg(['mean'])])
    st.write(pd.DataFrame(score_df.loc['mean']).T)


def _save_results(pipeline_info: PipelineInformation, data_info: DataInformation) -> None:
    import pickle
    pipeline = pipeline_info.pipeline

    @st.cache
    def save_model(pipeline):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return data_info.df.to_csv().encode('utf-8')

    st.download_button('Download Model', data=pickle.dumps(pipeline))


def _create_main_page(pipeline_info: PipelineInformation, data_info: DataInformation):
    left, right = st.columns(2)

    # problem overview
    with left:
        st.header('Problem')
        st.write(f'{"Classification Problem" if pipeline_info.classification else "Regression Problem "}')
        st.write(f'Data Set: {data_info.dataset_name}')
        st.write(f'Target Name: {data_info.target}')

    # plot pipeline
    with right:
        st.header('Pipeline overview')
        pipeline_diagram = estimator_html_repr(pipeline_info.pipeline)
        st.components.v1.html(pipeline_diagram, width=2000, height=200, scrolling=True)

    # configure training
    st.header('Customize Training and Cross Validation')

    # configure train test split schemes
    splitting_schemes = ['Random Shuffle Split'] # TODO: add more
    st.selectbox('Choose Splitting Scheme', splitting_schemes)
    test_size = st.number_input('Choose Test Size', min_value=0.05, max_value=0.95, value=0.3)

    # configure cross validation
    cv_schemes = ['K Fold'] # TODO: add more option
    st.selectbox('Choose CV Scheme', cv_schemes)
    num_folds = st.number_input('Choose # Folds', min_value=1, max_value=50, value=3)

    # load data
    X = data_info.df[data_info.features]
    y = data_info.df[data_info.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # train model
    pipeline_info.pipeline.fit(X_train, y_train)

    # predict
    y_hat = pipeline_info.pipeline.predict(X_test)

    _plot_results(
        pipeline_info=pipeline_info,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        y_hat=y_hat,
        num_folds=num_folds
    )

    _save_results(pipeline_info=pipeline_info, data_info=data_info)


def main():
    _init_header()
    data_info = _init_sidebar()
    pipeline_info = _build_pipeline(data_info=data_info)
    _create_main_page(pipeline_info=pipeline_info, data_info=data_info)


if __name__ == "__main__":
    main()