import pandas as pd
import numpy as np
import urllib
from astropy import units as u
from astropy.coordinates import SkyCoord


def download(type):
	"""
	Function for downloading the GZ1 dataset associated
	with the "type" csv file.

	To call:
		download(type)

	Parameters:
		type		"E", "Edge-on", or "S"

	Postcondtion:
		The files are stored in the folder specified
		by type as a jpeg image. The filenames are 
		the SDDS OBJID number.
	"""
	# ==================================================
	#	Open the csv file
	# ==================================================
	df = pd.read_csv(type + ".csv")

	# ==================================================
	#	Convert the coordinates to decimal
	# ==================================================
	coord = [a + " " + b for a, b in zip(df["RA"], df["DEC"])]
	coord = SkyCoord(coord, unit=(u.hour, u.deg), frame='icrs')

	# ==================================================
	#	Set the image properties and retrieve the url
	# ==================================================
	impix = 207
	scale = 0.5
	cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'

	for ra, dec, objid in zip(coord.ra.deg, coord.dec.deg, df["OBJID"].astype(str)):
		
		query_string = urllib.urlencode(dict(ra=ra, 
							dec=dec, 
							width=impix, 
							height=impix, 
							scale=scale))

		url = cutoutbaseurl + '?' + query_string

		# ==================================================
		#	Download the image
		# ==================================================	
		urllib.urlretrieve(url, type + "/" + objid + ".jpg")
