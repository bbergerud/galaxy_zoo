import numpy as np
import pandas as pd

df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
print(df.shape)

iloc_E = np.where(df["P_EL_DEBIASED"] == 1)[0]
iloc_S0 = np.where(df["P_EDGE"] > 0.95)[0]

iloc_S = np.where(
	((df["P_CW"] == 1) & (df["P_CS_DEBIASED"] == 1)) |
	((df["P_ACW"] == 1) & (df["P_CS_DEBIASED"] == 1))
	)[0]

E_id = df["OBJID"][iloc_E]
S0_id = df["OBJID"][iloc_S0]
S_id = df["OBJID"][iloc_S]

print(E_id.size)
print(S0_id.size)
print(S_id.size)
