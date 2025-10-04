import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
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
def model1(df,dataset_type,model_choice):
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
        if dataset_type == "tess":
            dfconfirmed = df[df[label_col].isin(pos_label)]
            dfFalsePositive = df[df[label_col].isin(neg_label)]
            dfcandidates = df[df[label_col].isin(candidate_label)]
        else:
            dfconfirmed = df[df[label_col]==pos_label]
            dfFalsePositive = df[df[label_col]==neg_label]
            dfcandidates = df[df[label_col]==candidate_label]
        dfcandidateposition = dfcandidates[position_cols]
        dfconfirmedfor3d = dfconfirmed[[label_col]+position_cols+[id_col]]
        dfconfirmed = dfconfirmed[[label_col]+feature_cols]
        dfFalsePositive = dfFalsePositive[[label_col]+feature_cols]
        dfcandidates = dfcandidates[[id_col,label_col]+feature_cols]
        dfconfirmed[label_col] = 1
        dfFalsePositive[label_col] = 0
        for col in feature_cols:
            if dfconfirmed[col].isnull().any():
                dfconfirmed[col].fillna(dfconfirmed[col].median(), inplace=True)
            if dfFalsePositive[col].isnull().any():
                dfFalsePositive[col].fillna(dfFalsePositive[col].median(), inplace=True)
        dfcombined = pd.concat([dfconfirmed, dfFalsePositive], ignore_index=True)
        dfcandidates = dfcandidates.drop(columns=[label_col])
        for col in feature_cols:
            if dfcandidates[col].isnull().any():
                dfcandidates[col].fillna(dfcombined[col].median(), inplace=True)
        X = dfcombined[feature_cols]
        y = dfcombined[label_col].astype(int)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
        scaler = StandardScaler()
        Xtrain_s = scaler.fit_transform(Xtrain)
        Xtest_s = scaler.transform(Xtest)
        if model_choice == "logreg":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == "rf":
            model = RandomForestClassifier(random_state=42)
        elif model_choice == "svm":
            model = SVC(probability=True, random_state=42)
        elif model_choice == "xgb":
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(Xtrain_s, ytrain)
        ypred = model.predict(Xtest_s)
        accuracy = accuracy_score(ytest, ypred)
        confmatrix = confusion_matrix(ytest, ypred)
        classreport = classification_report(ytest, ypred)
        coefs = model.coef_[0] if hasattr(model, "coef_") else []
        dfcandidates2 = dfcandidates.drop(columns=[id_col], errors='ignore')
        Xcand_s = scaler.transform(dfcandidates2)
        ycand_pred = model.predict(Xcand_s)
        ycand_predproba = model.predict_proba(Xcand_s)[:, 1]  
        dfcandidateswproba = dfcandidates.copy()
        dfcandidates[label_col] = ycand_pred
        dfcandidateswproba[label_col] =ycand_predproba
        results_df = pd.concat([dfcandidateposition, dfcandidates[[id_col,label_col]]], axis=1)
        results_dfwProba= pd.concat([dfcandidateposition,dfcandidateswproba[[id_col,label_col]]],axis=1)
        return results_df, accuracy, confmatrix, classreport, coefs, feature_cols, dfconfirmedfor3d, results_dfwProba
    

