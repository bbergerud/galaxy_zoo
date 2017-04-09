import pandas as pd
import numpy as np
import urllib, sys, os
from astropy import units as u
from astropy.coordinates import SkyCoord

def download(type, nPetro=1.5, impix=299):
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
	# ==================================================
	#	Open the csv file
	# ==================================================
	df = pd.read_csv('csv/' + type + ".csv")

	# ==================================================
	#	Test to see if the directories for placing
	#	the images exist. If not, create directory
	# ==================================================
	groups = set(df['group'])
	for group in groups:
		dir = group + '/' + type
		if not os.path.exists(dir):
			os.makedirs(dir)

	# ==================================================
	#	Set the image properties and retrieve the url
	#	Set the total number of arcseconds equal to
	#	2x the desired petrosian radius (diameter)
	# ==================================================
	narc = 2. * nPetro * df["petroRad_g"].values
	scales = narc / impix

	cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'

	for ra, dec, objid, group, scale in zip(df['ra'], df['dec'], df["objid"].astype(str), df["group"], scales):
		
		query_string = urllib.urlencode(dict(ra=ra, 
							dec=dec, 
							width=impix, 
							height=impix, 
							scale=scale))

		url = cutoutbaseurl + '?' + query_string

		# ==================================================
		#	Download the image
		# ==================================================
		urllib.urlretrieve(url, group + "/" + type + "/" + objid + ".jpg")
