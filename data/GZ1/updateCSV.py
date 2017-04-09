import numpy as np
import pandas as pd
import mechanize
from StringIO import StringIO
url = 'http://cas.sdss.org/DR7/en/tools/search/sql.asp'


def updateCSV(type, trainNum=2200, validNum=800):

	# ==================================================
	#	Open the CSV file and get the OBJIDs
	# ==================================================
	df = pd.read_csv(type + '.csv')
	df = df.sort_values(by='OBJID', axis=0)

	# ==================================================
	#	Create the SQL query to extract the petrosian
	#	radius from the database
	# ==================================================
	s = '''
		SELECT objid, ra, dec, petroRad_g
		FROM PhotoObj
		WHERE
		'''

	for objid in df["OBJID"]:
		s += ' objid = {:d} OR '.format(objid)
	s = s[:-3]

	# ==================================================
	#	Connect to the SDSS webpage and query the
	#	database
	# ==================================================
	br = mechanize.Browser()
	resp = br.open(url)

	br.select_form(name='sql')
	br['cmd'] = s
	br['format'] = ['csv']
	response = br.submit()

	# ==================================================
	#	Turn the reponse into a data frame
	# ==================================================
	file_like = StringIO(response.get_data())
	df = pd.read_csv(file_like)

	# =========================================
	#	Partition the dataset into testing,
	#	training, and validation
	# =========================================
	df["group"] = 'test'
	loc = np.random.choice(df.index, replace=False, size=trainNum)
	df.ix[loc, 'group'] = 'train'
	loc = np.random.choice(df.index[df["group"] != 'train'], replace=False, size=validNum)
	df.ix[loc, 'group'] = 'valid'

	# =========================================
	#	Save the dataframe as a csv file
	# =========================================

	df.to_csv('csv/' + type + '.csv')
