import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score


def my_scores(df_g, roc=True):
  d={}
  d["acc"] = accuracy_score(y_true=df_g["y"],y_pred=df_g["predicted_label"])
  d["N"] = df_g.shape[0]

  if roc:
      d["AUC"] = roc_auc_score(y_true=df_g["y"],y_score=df_g["pred_score_1_soft"])
      return pd.Series(d,index=["acc","AUC","N"])
  else:
      return pd.Series(d,index=["acc","N"])

def weighted_avg(df_g):
  d={}
  d["acc"]=np.average(df_g["acc"],weights=df_g["N"])
  d["+/-(acc)"]=np.sqrt(np.average(
      (df_g["acc"]-d["acc"])**2,
      weights=df_g["N"]
      ))
  d["AUC"]=np.average(df_g["AUC"],weights=df_g["N"])
  d["+/-(AUC)"]=np.sqrt(np.average(
      (df_g["AUC"]-d["AUC"])**2,
      weights=df_g["N"]
      ))
  return pd.Series(d,index=["acc","+/-(acc)","AUC","+/-(AUC)"])
