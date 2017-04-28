import pandas as pd
import numpy as np
import sys, os
import urllib.parse
import urllib.request
from astropy import units as u
from astropy.coordinates import SkyCoord

def download(type, nPetro=1.75, impix=299, nprint=500):
	"""
	Function for downloading the GZ1 dataset associated
	with the "type" csv file.

	To call:
		download(type, nPetro, impix)

	Parameters:
		type		"E", "S0", or "S"
		nPetro		number of Pertrosian radii
		impix		image size (impix X impix)

	Postcondtion:
		The files are stored in the folder specified
		by type as a jpeg image. The filenames are 
		the SDDS OBJID number.
	"""

	for folder in ['csv2', 'csvE']:

		# ==================================================
		#	Open the csv file
		# ==================================================
		df = pd.read_csv(folder + '/' + type + ".csv")

		# ==================================================
		#	Test to see if the directories for placing
		#	the images exist. If not, create directory
		# ==================================================
		groups = set(df['group'])
		for group in groups:
			dir = 'data/' + group + '/' + type
			if not os.path.exists(dir):
				os.makedirs(dir)

		# ==================================================
		#	Set the image properties and retrieve the url
		#	Set the total number of arcseconds equal to
		#	2x the desired petrosian radius (diameter)
		# ==================================================
		narc = 2. * nPetro * df["petroR90_g"].values
		scales = narc / impix

		cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'


		N = df.shape[0]
		for i, (ra, dec, objid, group, scale) in enumerate(zip(df['ra'], df['dec'], df["objid"].astype(str), df["group"], scales)):

			# ==================================================
			#	Print the progress
			# ==================================================
			if i % nprint == 0:
				print("Iter #{:d} / {:d}".format(i, N))
		

			# ==================================================
			#	Create the query string
			# ==================================================
			query_string = urllib.parse.urlencode(dict(ra=ra, 
								dec=dec, 
								width=impix, 
								height=impix, 
								scale=scale))

			url = cutoutbaseurl + '?' + query_string

			# ==================================================
			#	Download the image
			# ==================================================
			urllib.request.urlretrieve(url, "data/" + group + "/" + type + "/" + objid + ".jpg")
