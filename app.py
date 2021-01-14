import streamlit as st

# EDA packages

import pandas as pd
import numpy as np

# utils packages

import os
import joblib

# Data viz packages

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from plot_tsne_py import PlotTSNE

# matplotlib.use('Agg')

def main():
    """ TB DASHBOARD """

    st.title("TB DASHBOARD")

    menu = ["Home", "Login", "Signup"]
    choice1 = ["Metrics1", "Metrics2"]
    choice2 = ["Model1", "Model2"]

    # Dropdowns
    algo = st.sidebar.selectbox("Select Algorithm", ("Algo1", "Algo2", "Algo3", "Algo4", "Algo5"))
    metrics = st.sidebar.selectbox("Choose Metrics", choice1)
    activity = st.sidebar.selectbox("Choose pre-trained models", choice2)

    # Text input 
    text = st.sidebar.text_input('Input your genes here:')

    # File uploader
    data_file = st.file_uploader("Upload CSV", type = ["csv"])

    if st.button("Process"):
        if data_file is not None:
            st.write(type(data_file))
            #df = pd.read_csv(data_file)
            #st.dataframe(df)
            rd = PlotTSNE(data_file, "/Users/mypc/Desktop/streamlit/gpbt2Axes_1000.csv")
            rd.plot_tsne()



if __name__ == '__main__':
    main()
