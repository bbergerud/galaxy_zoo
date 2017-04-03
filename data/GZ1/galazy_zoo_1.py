import numpy as np
import pandas as pd

df = pd.read_csv("GalaxyZoo1_DR_table2.csv")
print(df.shape)

E  = df.loc[df["P_EL_DEBIASED"] == 1, ["OBJID", "RA", "DEC"]]
S0 = df.loc[df["P_EDGE"] > 0.95, ["OBJID", "RA", "DEC"]]
S  = df.loc[	((df["P_CW"] == 1) & (df["P_CS_DEBIASED"] == 1)) 	|
				((df["P_ACW"] == 1) & (df["P_CS_DEBIASED"] == 1)), ["OBJID", "RA", "DEC"]	]

print(E.shape)
print(S0.shape)
print(S.shape)

E.to_csv("E.csv", index=False)
S0.to_csv("Edge-on.csv", index=False)
S.to_csv("S.csv", index=False)
