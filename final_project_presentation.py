import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import functions as f
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:,.2f}'.format 

st.title("OLYMPIC DATA ANALYSIS")

############### LOAD DATA ###############

olympic_df = pd.read_csv('dataset_olympic_data/dataset_olympics.csv')
noc_region_df = pd.read_csv('dataset_olympic_data/noc_region.csv')

############### EXPLORE THA DATASET ###############

with st.sidebar:
    add_menu = st.radio(
        "test",
        ("First", "Second")
    )
