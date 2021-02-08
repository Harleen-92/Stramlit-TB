import streamlit as st

# EDA packages

import pandas as pd
import numpy as np

# utils packages

import os
import joblib
import base64

# Data viz packages

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from plot_tsne_py import PlotTSNE
from generate import generate
from generate import Generator
from generate import Discriminator

# matplotlib.use('Agg')

hide_streamlit_style = """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# File Download

def make_downloadable(df, format_type='csv'):
    if format_type == 'csv':
        datafile = df.to_csv(index = 'False')
    elif format_type =='json':
        datafile = df.to_json()

    b64 = base64.b64encode(datafile.encode()).decode()
    href = f'<a href="data:file/{format_type};base64,{b64}" download="myfile.csv">Download file</a>'
    st.markdown(href, unsafe_allow_html=True)


def main():
    """ TB DASHBOARD """

    st.title("TB DASHBOARD")

    #menu = ["Home", "Login", "Signup"]
    #choice1 = ["Metrics1", "Metrics2"]
    choice1 = ["Select model", "wgan", "gmm"]
    choice2 = ["Select datatype", "TB", "healthy"]

    # Data generation

    models = st.sidebar.selectbox("Choose pre-trained models", choice1)
    datatype = st.sidebar.selectbox("Choose pre-trained models", choice2)
    number = st.sidebar.number_input("Number", 10, 1000)
    dataformat = st.sidebar.selectbox("Save As", ['csv', 'json'])

    if st.sidebar.button("Generate Samples"):
        df = generate(models, number, datatype)
        st.dataframe(df)
        make_downloadable(df, dataformat)

    # Text input
    #text = st.sidebar.text_input('Input your genes here:')


if __name__ == '__main__':
    main()
