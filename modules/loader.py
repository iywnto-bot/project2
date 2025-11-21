# modules/loader.py
import pandas as pd
import pickle

def load_data():
    df = pd.read_csv("data/cho_tot_cleaned_wt.csv")
    return df

def load_data_cluster():
    df_cluster = pd.read_csv("data/cho_tot_cluster.csv")
    return df_cluster

def load_model():
    with open("data/xe_cosine_sim.pkl", "rb") as f:
        model = pickle.load(f)
    return model
