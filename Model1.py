import pandas as pd
import numpy as np
def add_kepler_distance(df):
    sigma = 5.670374419e-8
    R_sun = 6.957e8
    pc = 3.086e16
    F0 = 3.6e-8
    df = df.copy()
    df["koi_srad"] = df["koi_srad"].astype(float).fillna(df["koi_srad"].median())
    df["koi_steff"] = df["koi_steff"].astype(float).fillna(df["koi_steff"].median())
    df["koi_kepmag"] = df["koi_kepmag"].astype(float).fillna(df["koi_kepmag"].median())
    R_star = df["koi_srad"].values * R_sun
    T_star = df["koi_steff"].values
    m_kep = df["koi_kepmag"].values

    L_star = 4 * np.pi * (R_star**2) * sigma * (T_star**4)
    F_obs = F0 * 10**(-0.4 * m_kep)
    d_m = np.sqrt(L_star / (4 * np.pi * F_obs))
    df["koi_dist"] = d_m / pc
    return df
def model1(df,dataset_type):
        if dataset_type == "k2":
            label_col = "disposition"
            id_col = "pl_name"
            feature_cols = ['pl_orbper','sy_dist','sy_vmag','sy_kmag','sy_gaiamag']
            position_cols = ['ra','dec','sy_dist']
            pos_label = "CONFIRMED"
            neg_label = "FALSE POSITIVE"
            candidate_label = "CANDIDATE"
        elif dataset_type == "kepler":
            df=add_kepler_distance(df)
            label_col = "koi_disposition"
            id_col = "kepid"
            feature_cols = ['koi_period','koi_impact','koi_duration','koi_depth','koi_prad','koi_teq','koi_slogg','koi_srad']
            position_cols = ['ra','dec','koi_dist']
            pos_label = "CONFIRMED"
            neg_label = "FALSE POSITIVE"
            candidate_label = "CANDIDATE"
        elif dataset_type == "tess":
            label_col = "tfopwg_disp"
            id_col = "toi"
            feature_cols = ['st_pmra','pl_trandurh','pl_trandep','pl_rade','pl_insol','pl_eqt','st_tmag','st_dist','st_teff']
            position_cols = ['ra','dec','st_dist']   
            pos_label = ["CP","KP"]
            neg_label = ["FP"]
            candidate_label = ["PC","APC"]