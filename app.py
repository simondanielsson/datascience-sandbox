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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

open_ml_dataset_names = ['titanic', 'credit-g', 'liver-disorders', 'cholesterol', 'spambase', 'bodyfat']

st.title('Supervised Learning Sand-Box')

st.sidebar.title('Create DS Problem')
OPEN_ML_DS_NAME = st.sidebar.selectbox('Select A Toy Data Set', open_ml_dataset_names)

uploaded_file = st.sidebar.file_uploader('Or upload your own')

df = sklearn_to_df(fetch_openml(OPEN_ML_DS_NAME, version=1, as_frame=True, return_X_y=False))
if uploaded_file is not None:
#read csv
    df = pd.read_csv(uploaded_file)


try:
    default_index = list(df.columns).index('target')
except:
    default_index = 0

TARGET_NAME = st.sidebar.selectbox('Choose Target Name', list(df.columns), default_index)
FEATURE_NAMES = list(df.columns)
FEATURE_NAMES.remove(TARGET_NAME)
FEATURE_NAMES = st.sidebar.multiselect('Choose Features', options = FEATURE_NAMES, default = FEATURE_NAMES)

numeric_feature_names = list(df[FEATURE_NAMES].select_dtypes('number').columns)
categoric_feature_names = list( set(df[FEATURE_NAMES].columns) - set(numeric_feature_names))
N_levels = 15
is_classification = len(df[TARGET_NAME].unique()) <= N_levels

st.sidebar.header('Create Model')
estimator_name = st.sidebar.selectbox('Choose Estimator', ['Linear Regression', 'Ridge'])
x = st.sidebar.selectbox('ADD MORE OPTIONS HERE', ['Linear Regression', 'Ridge'])


from sklearn import set_config
set_config(display="diagram")
imputer = SimpleImputer()
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy = 'mean')),
                                ('scaler', StandardScaler())])
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy = 'most_frequent')),
                                    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))])
preprocessor = ColumnTransformer(transformers = [('numeric', numeric_transformer, numeric_feature_names),
                                                 ('categoric', categorical_transformer, categoric_feature_names)]
                                )

estimator = LogisticRegression() if is_classification else LinearRegression()

from sklearn.utils import estimator_html_repr

pipe = Pipeline(steps = [('preprocessing', preprocessor), ('estimator' , estimator)])



left, right = st.columns(2)

with left:
    st.header('Problem')
    st.write(f'{"Regression Problem " if not is_classification else "Classification Problem"}')
    st.write(f'Data Set: {OPEN_ML_DS_NAME}')
    st.write(f'Target Name: {TARGET_NAME}')
with right:
    st.header('Model')
    pipe_diag = estimator_html_repr(pipe)
    st.components.v1.html(pipe_diag, width=2000, height=200, scrolling=True)


st.header('Customize Training and Cross Validation')
X = df[FEATURE_NAMES]
y = df[TARGET_NAME]
st.selectbox('Choose Splitting Scheme', ['Random Shuffle Split'])
test_size = st.number_input('Choose Test Size', min_value = 0.05, max_value = 0.95, value = 0.3)
st.selectbox('Choose CV Scheme', ['K Fold'])
num_folds = st.number_input('Choose # Folds', min_value = 1, max_value = 50, value = 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

pipe.fit(X_train, y_train)
y_hat = pipe.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
st.header('Results')
st.set_option('deprecation.showPyplotGlobalUse', False)
left, right = st.columns(2)
with right:
    
    if is_classification and len(y_test.unique()) <= 2:
        st.header('ROC Curve')
        fig_1 = RocCurveDisplay.from_estimator(pipe, X_test, y_test).plot()
        st.pyplot()
with left:

    if is_classification:
        st.header('CNF Matrix')
        cm = confusion_matrix(y_test, y_hat, labels = pipe.named_steps['estimator'].classes_)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.named_steps['estimator'].classes_).plot()
        st.pyplot()

from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro'] if is_classification else ['r2', 'explained_variance',
                                                                         'neg_mean_absolute_error']
from sklearn.model_selection import cross_validate
scores = cross_validate(pipe, X, y, scoring=scoring, cv = num_folds)
score_df = pd.DataFrame(scores)
score_df = pd.concat([score_df, score_df.agg(['mean'])]) 
st.write(pd.DataFrame(score_df.loc['mean']).T)

import pickle
@st.cache
def save_model(pipe):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



st.download_button('Download Model', data = pickle.dumps(pipe))







