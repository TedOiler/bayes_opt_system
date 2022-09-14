import numpy as np
import streamlit as st
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from helpers_scikit_learn import *
from datetime import datetime
from matplotlib import pyplot as plt
import rdkit as rd
from scipy.spatial import distance
from rdkit.Chem import Descriptors
import math

plt.style.use('fivethirtyeight')

if 'exp_number' not in st.session_state:
    st.session_state.exp_number = 0


@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')


@st.cache
def smiles_to_desc(s):
    m = rd.Chem.MolFromSmiles(s)
    molweight_desc = Descriptors.MolWt(m)
    rotatable_desc = Descriptors.NumRotatableBonds(m)
    logp_desc = Descriptors.MolLogP(m)  # less reliable
    qed_desc = Descriptors.qed(m) * 1000
    # return [molweight_desc, rotatable_desc, logp_desc, qed_desc]
    return [molweight_desc, qed_desc]


@st.cache
def desc_to_smiles(d, p):
    dsts = pd.DataFrame(
        [distance.euclidean(d.iloc[i, 1:], p) for i in range(d.shape[0])]
    )
    dsts.columns = ['distance_ligands']
    d = pd.concat([d, dsts], axis=1)
    m = d['ligands'][d['distance_ligands'].argmin()]
    return m


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
    "Probability of Improvement": acq_pi,
    "Upper Confidence Bound": acq_ucb,
}

# Sidebar --------------------------------------------------------------------------------------------------------------
input_form = st.sidebar.form(key="input")
input_expander = input_form.expander("Initial Data")

with input_expander:
    uploaded_file = input_expander.file_uploader("Random Start", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=[0])
        except:
            df = pd.read_csv(uploaded_file, header=[0])
    else:
        df = None
        latent_space = None

    ligand_file = input_expander.file_uploader("Ligands List", type=['xlsx', 'csv'])
    if ligand_file is not None:
        try:
            ligand_df = pd.read_excel(ligand_file, header=[0])
            ligand_desc = pd.DataFrame([smiles_to_desc(ligand_df['ligands'][i]) for i in range(ligand_df.shape[0])])
            ligand_full_data = pd.concat([ligand_df, ligand_desc], axis=1)
            ligand_full_data.columns = [ligand_df.columns[0], f'mol_weight_{ligand_df.columns[0]}',
                                        f'qed_{ligand_df.columns[0]}']

            ligand_min_mol_weight = ligand_full_data[f'mol_weight_{ligand_df.columns[0]}'].min()
            ligand_max_mol_weight = ligand_full_data[f'mol_weight_{ligand_df.columns[0]}'].max()

            # ligand_min_rotatable = ligand_full_data[f'rotatable_{ligand_df.columns[0]}'].min()
            # ligand_max_rotatable = ligand_full_data[f'rotatable_{ligand_df.columns[0]}'].max()

            # ligand_min_logp = ligand_full_data[f'logp_{ligand_df.columns[0]}'].min()
            # ligand_max_logp = ligand_full_data[f'logp_{ligand_df.columns[0]}'].max()

            ligand_min_qed = ligand_full_data[f'qed_{ligand_df.columns[0]}'].min()
            ligand_max_qed = ligand_full_data[f'qed_{ligand_df.columns[0]}'].max()

        except:
            ligand_df = pd.read_csv(ligand_file, header=[0])
            ligand_desc = pd.DataFrame([smiles_to_desc(ligand_df['ligands'][i]) for i in range(ligand_df.shape[0])])
            ligand_full_data = pd.concat([ligand_df, ligand_desc], axis=1)
            ligand_full_data.columns = [ligand_df.columns[0], f'mol_weight_{ligand_df.columns[0]}',
                                        f'qed_{ligand_df.columns[0]}']

            ligand_min_mol_weight = ligand_full_data[f'mol_weight_{ligand_df.columns[0]}'].min()
            ligand_max_mol_weight = ligand_full_data[f'mol_weight_{ligand_df.columns[0]}'].max()

            # ligand_min_rotatable = ligand_full_data[f'rotatable_{ligand_df.columns[0]}'].min()
            # ligand_max_rotatable = ligand_full_data[f'rotatable_{ligand_df.columns[0]}'].max()

            # ligand_min_logp = ligand_full_data[f'logp_{ligand_df.columns[0]}'].min()
            # ligand_max_logp = ligand_full_data[f'logp_{ligand_df.columns[0]}'].max()

            ligand_min_qed = ligand_full_data[f'qed_{ligand_df.columns[0]}'].min()
            ligand_max_qed = ligand_full_data[f'qed_{ligand_df.columns[0]}'].max()
    else:
        ligand_df = None
        ligand_desc = None
        ligand_full_data = None

submit_data_btn = input_form.form_submit_button("✅ Submit")

# Main -----------------------------------------------------------------------------------------------------------------
st.write(
    '<img width=100 src="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/325/robot_1f916.png" style="margin-left: 5px; brightness(1);">',
    unsafe_allow_html=True)
st.title("Bayesian Optimization Application")

# Data -----------------------------------------------------------------------------------------------------------------
parameter_space_expander = st.expander("Parameter Space")
parameter_space_tab1, parameter_space_tab2, parameter_space_tab3 = parameter_space_expander.tabs(
    ["Parameter Input", "Parameter Output", "Ligand"])

latent_space_expander = st.expander("Latent Space")
latent_space_tab1, latent_space_tab2 = latent_space_expander.tabs(
    ["Latent Input", "Latent Output"]
)
if df is not None:

    ligand_desc_latent = pd.DataFrame([smiles_to_desc(df['Ligand_SMILES'][i]) for i in range(df.shape[0])])
    ligand_desc_latent.columns = [f'mol_weight_{df.columns[1]}', f'qed_{df.columns[1]}']

    latent_space = pd.concat([
        ligand_desc_latent,
        df.iloc[:, 1:]
    ], axis=1)

    parameter_space_tab1.table(df)
    parameter_space_tab1.download_button(label="📥 Export random start data to CSV", data=convert_df(df),
                                         file_name=f'random_start_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                         mime='text/csv')

    parameter_space_tab3.table(ligand_full_data)
    parameter_space_tab3.download_button(label="📥 Export ligands data to CSV", data=convert_df(ligand_df),
                                         file_name=f'ligands_list_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                         mime='text/csv')

    latent_space_tab1.table(
        latent_space
    )
    latent_space_tab1.download_button(label="📥 Export random start latent space data to CSV", data=convert_df(df),
                                      file_name=f'random_start_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                      mime='text/csv')

    st.sidebar.write("---")
    if round(df.iloc[-2, -1], 4) == 0:
        local_prc_change = 100
    else:
        local_prc_change = np.round(((df.iloc[-1, -1] - df.iloc[-2, -1]) / df.iloc[-2, -1]) * 100, 2)
    local_abs_change = np.round(df.iloc[-1, -1] - df.iloc[-2, -1], 2)
    st.sidebar.write("### Metrics")
    metrics_tab1, metrics_tab2 = st.sidebar.tabs(["Current vs Previous", " Current vs Global"])
    col1, col2 = metrics_tab1.columns(2)
    col1.metric("Percentage", value=np.round(df.iloc[-1, -1], 2), delta=f'{local_prc_change}%')
    col2.metric("Absolute", value=np.round(df.iloc[-1, -1], 2), delta=f'{local_abs_change}')
    col11, col22 = metrics_tab2.columns(2)
    if round(df.iloc[-1, -1], 4) == 0:
        global_prc_change = 100
    else:
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
# low = [ligand_min_mol_weight, ligand_min_qed]
low = [138.149, 0.124*1000]
# high = [ligand_max_mol_weight, ligand_max_qed]
high = [796.673, 0.741*1000]
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
beta = experiment_params_form.slider("beta", 0.1, 1.0, 0.5, 0.01)
experiment_params_form.write("---")
nexp = 1
model = experiment_params_form.selectbox("Surrogate Model:", ["Gaussian Process"])
acq = experiment_params_form.selectbox("Acquisition Function:",
                                       ["Expected Improvement", "Probability of Improvement", "Upper Confidence Bound"])
experiment_params_form_btn = experiment_params_form.form_submit_button("✅ Submit")

if experiment_params_form_btn:
    st.session_state.exp_number += 1
    with st.spinner("Training model"):
        X = latent_space.iloc[:, :-1]
        y = latent_space.iloc[:, -1]
        model_dict[model].fit(X, y)
        next_x = opt_acq(X=X,
                         y=y,
                         model=model_dict[model],
                         acq=acq_dict[acq],
                         low=low,
                         high=high,
                         beta=beta)

        next_x_parameter = pd.DataFrame([desc_to_smiles(ligand_full_data, next_x[:2]), *list(next_x[2:]), math.nan]).T
        next_x_parameter.columns = df.columns
        output_df_parameter = df.append(next_x_parameter, ignore_index=True)
        parameter_space_tab2.table(output_df_parameter)
        parameter_space_tab2.download_button(label="📥 Export output data to CSV", data=convert_df(output_df_parameter),
                                          file_name=f'output_data_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                          mime='text/csv')

        next_x_latent = pd.DataFrame(np.append(next_x, np.nan)).T
        next_x_latent.columns = latent_space.columns
        output_df_latenet = latent_space.append(next_x_latent, ignore_index=True)
        latent_space_tab2.table(output_df_latenet)
        latent_space_tab2.download_button(label="📥 Export latent output data to CSV", data=convert_df(output_df_latenet),
                                          file_name=f'output_data_{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv',
                                          mime='text/csv')


    experiment_params_expander.success(
        f'Model number {st.session_state.exp_number} Successfully Built! (Check Output Input Data Tab)')
