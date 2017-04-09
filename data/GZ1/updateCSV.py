import numpy as np
import pandas as pd
import mechanize
from StringIO import StringIO
url = 'http://cas.sdss.org/DR7/en/tools/search/sql.asp'


def updateCSV(type):
	df = pd.read_csv(type + '.csv')
	df = df.sort_values(by='OBJID', axis=0)

	s = '''
		SELECT objid, ra, dec, petroRad_g
		FROM PhotoObj
		WHERE
		'''

	for objid in df["OBJID"]:
		s += ' objid = {:d} OR '.format(objid)
	s = s[:-3]

	br = mechanize.Browser()
	resp = br.open(url)

	br.select_form(name='sql')
	br['cmd'] = s
	br['format'] = ['csv']
	response = br.submit()

	file_like = StringIO(response.get_data())
	df = pd.read_csv(file_like)
	df.to_csv(type + '_scale.csv')
