import pandas as pd
import numpy 
def model1(df,dataset_type):
        if dataset_type == "k2":
            label_col = "disposition"
            id_col = "pl_name"
            feature_cols = ['pl_orbper','sy_dist','sy_vmag','sy_kmag','sy_gaiamag']
            Position = ['ra','dec','sy_dist']
            pos_label = "CONFIRMED"
            neg_label = "FALSE POSITIVE"
            candidate_label = "CANDIDATE"
        elif dataset_type == "kepler":
            label_col = "koi_disposition"
            id_col = "kepid"
            feature_cols = ['koi_period','koi_impact','koi_duration','koi_depth','koi_prad','koi_teq','koi_slogg','koi_srad']
            Position = ['ra','dec']
            pos_label = "CONFIRMED"
            neg_label = "FALSE POSITIVE"
            candidate_label = "CANDIDATE"
        elif dataset_type == "tess":
            label_col = "tfopwg_disp"
            id_col = "toi"
            feature_cols = ['st_pmra','pl_trandurh','pl_trandep','pl_rade','pl_insol','pl_eqt','st_tmag','st_dist','st_teff']
            Position = ['ra','dec','st_dist']   
            pos_label = ["CP","KP"]
            neg_label = ["FP"]
            candidate_label = ["PC","APC"]