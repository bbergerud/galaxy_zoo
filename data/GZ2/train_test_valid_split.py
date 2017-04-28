import glob
import pandas as pd
import numpy as np

csv_files = glob.glob('*.csv')

"""
for csv in csv_files:
	df = pd.read_csv(csv)

	df['group'] = 'train'

	# ===================================
	#	Send 20% to the test dataset
	# ===================================
	size = int(0.2 * df.group.shape[0])

	loc = np.random.choice(df.index, replace=False, size=size)
	df.loc[loc, 'group'] = 'test'

	# ===================================
	#	Send 20% to the valid dataset
	# ===================================
	loc = np.random.choice(df[df['group'] != 'test'].index, replace=False, size=size)
	df.loc[loc, 'group'] = 'valid'

	df = df.loc[:,['dr7objid', 'ra', 'dec', 'hubble', 'group']]
	df.columns.values[0] = "objid"
	#df.drop(df.columns[[3]], axis=1, inplace=True)
	df.to_csv('csv/' + csv, index=False)
"""

def merge(ttype):
	csv_files = glob.glob('csv/' + ttype + '*.csv')

	for csv in csv_files:
		try:
			df
		except:
			df = pd.read_csv(csv)
			continue

		df_cur = pd.read_csv(csv)
		df = pd.concat([df, df_cur])

	return df
