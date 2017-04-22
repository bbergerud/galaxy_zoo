import numpy as np
import pandas as pd
import mechanize, glob
from StringIO import StringIO
url = 'http://cas.sdss.org/DR7/en/tools/search/sql.asp'


def getSize(type):

	# ==================================================
	#	Open the CSV file and get the OBJIDs
	# ==================================================
	df = pd.read_csv('csv/' + type + '.csv')
	df = df.sort_values(by='objid', axis=0)

	# ==================================================
	#	Create the SQL query to extract the petrosian
	#	radius from the database
	# ==================================================
	s = '''
		SELECT objid, ra, dec, petroR90_g
		FROM PhotoObj
		WHERE
		'''

	for objid in df["objid"]:
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
	df2 = pd.read_csv(file_like)

	df2 = df2.sort_values(by='objid', axis=0)
	df2['group'] = df['group']
	df2['hubble'] = df['hubble']


	# =========================================
	#	Save the dataframe as a csv file
	# =========================================

	df2.to_csv('csv2/' + type + '.csv', index=False)


files = glob.glob('csv/*')

for fn in files:
	fn = fn.split('/')[1]
	fn = fn.split('.')[0]
	getSize(fn)
