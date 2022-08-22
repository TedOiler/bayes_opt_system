import numpy as np
import streamlit as st
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from scikit_learn_helpers import *
from datetime import datetime
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


st.set_page_config(
    page_title="BO Dashboard",
    page_icon="📈",
    layout="wide",
)

model_dict = {
    "Gaussian Process": GaussianProcessRegressor()
}

acq_dict = {
    "Expected Improvement": acq_ei,
    "Probability of Improvement": acq_pi
}

# Sidebar --------------------------------------------------------------------------------------------------------------
input_form = st.sidebar.form(key="input")
input_expander = input_form.expander("Initial Data")

with input_expander:
    uploaded_file = input_expander.file_uploader("Choose a file", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=[0])
        except:
            df = pd.read_csv(uploaded_file, header=[0])
    else:
        df = None
submit_data_btn = input_form.form_submit_button("✅ Submit")

# Main -----------------------------------------------------------------------------------------------------------------
st.write(
    '<img width=100 src="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/robot_1f916.png" style="margin-left: 5px; brightness(1);">',
    unsafe_allow_html=True)
st.title("Bayesian Optimization Application")

# Data -----------------------------------------------------------------------------------------------------------------
data_expander = st.expander("Data")
data_tab1, data_tab2 = data_expander.tabs(["Input Data", "Output Data"])
if df is not None:
    data_tab1.table(df)
    data_tab1.download_button(label="📥 Export input data to CSV", data=convert_df(df),
                              file_name=f'input_data_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                              mime='text/csv')
    st.sidebar.write("---")
    local_prc_change = np.round(((df.iloc[-1, -1] - df.iloc[-2, -1]) / df.iloc[-2, -1]) * 100, 2)
    local_abs_change = np.round(df.iloc[-1, -1] - df.iloc[-2, -1], 2)
    st.sidebar.write("### Metrics")
    metrics_tab1, metrics_tab2 = st.sidebar.tabs(["Current vs Previous", " Current vs Global"])
    col1, col2 = metrics_tab1.columns(2)
    col1.metric("Percentage", value=np.round(df.iloc[-1, -1], 2), delta=f'{local_prc_change}%')
    col2.metric("Absolute", value=np.round(df.iloc[-1, -1], 2), delta=f'{local_abs_change}')
    col11, col22 = metrics_tab2.columns(2)
    global_prc_change = np.round(((df.iloc[-1, -1] - df.iloc[:, -1].max()) / df.iloc[-1, -1]) * 100, 2)
    global_abs_change = np.round(df.iloc[-1, -1] - df.iloc[:, -1].max(), 2)
    col11.metric("Percentage", value=np.round(df.iloc[:, -1].max(), 2), delta=f'{global_prc_change}%')
    col22.metric("Absolute", value=np.round(df.iloc[:, -1].max(), 2), delta=f'{global_abs_change}')
    st.sidebar.write("---")
    st.sidebar.write("### Timeline of yield")
    z = np.polyfit(df.index, df.iloc[:, -1], 2)
    p = np.poly1d(z)
    fig = plt.figure()
    plt.plot(df.iloc[:, -1])
    plt.plot(df.index, p(df.index), "r--")
    st.sidebar.pyplot(fig)
    st.sidebar.write("---")
st.write("---")

# Parameters -----------------------------------------------------------------------------------------------------------
low = []
high = []
experiment_params_expander = st.expander("Process Parameters")
experiment_params_form = experiment_params_expander.form(key="param")

for i in range(2):
    col1, col2 = experiment_params_form.columns(2)
    low_i = col1.number_input(label=f"low {i + 1}")
    high_i = col2.number_input(label=f"high {i + 1}")
    low.append(low_i)
    high.append(high_i)
if any(v == 0 for v in list(np.subtract(np.array(low), np.array(high)))):
    experiment_params_form.warning("High and Low values cannot be equal")
experiment_params_form.write("---")
nexp = 1
model = experiment_params_form.selectbox("Surrogate Model:", ["Gaussian Process"])
acq = experiment_params_form.selectbox("Acquisition Function:", ["Expected Improvement", "Probability of Improvement"])
experiment_params_form_btn = experiment_params_form.form_submit_button("✅ Submit")

if experiment_params_form_btn:
    with st.spinner("Training model"):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model_dict[model].fit(X, y)
        next_x = opt_acq(X=X,
                         y=y,
                         model=model_dict[model],
                         acq=acq_dict[acq],
                         low=low,
                         high=high)
        next_x_df = pd.DataFrame(np.append(next_x, np.nan)).T
        next_x_df.columns = df.columns
        output_df = df.append(next_x_df, ignore_index=True)
        data_tab2.table(output_df)
        data_tab2.download_button(label="📥 Export output data to CSV", data=convert_df(output_df),
                                  file_name=f'output_data_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                  mime='text/csv')
    experiment_params_expander.success('Model Successfully Built! (Check Output Data Tab)')
