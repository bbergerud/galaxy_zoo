import glob
import numpy as np

from inception import *

from keras.preprocessing.image import ImageDataGenerator

class hubble:
	
	def __init__(self, batch_size=32, l2reg=0.0002):
		self.__batchSize = batch_size
		self.__l2reg = l2reg


	def _setTrain(self, target, class_weight=False):
		"""
		Function for preparing the training and
		validation testsets.

		To call:
			_setTrain(target, weights=False)

		Parameters:
			target		target directory
			class_weight	(boolean) apply class weighting

		Postcondition:
			The ImageDataGenerators have been stored
			in the variables imgTrain and imgValid.

			The class counts for the training and
			validation sets have been stored in the
			dictionary imgCounts.

			The class weights have been stored in the
			variable imgWeights.
		"""
		# =======================================
		# Set the train and valid directories
		# =======================================
		if target[-1] == '/':
			trainDir = target + 'train/'
			validDir = target + 'valid/'
		else:
			trainDir = target + '/train/'
			validDir = target + '/valid/'


		# =======================================
		# Create the image generator
		# =======================================
		datagen = ImageDataGenerator()

		self.imgTrain = datagen.flow_from_directory(
			directory = trainDir,
			batch_size = self.__batchSize,
			class_mode = 'categorical')

		self.imgValid = datagen.flow_from_directory(
			directory = validDir,
			batch_size = self.__batchSize,
			class_mode = 'categorical')

		trainCounts = len(glob.glob(trainDir + '*/*jpg'))
		validCounts = len(glob.glob(validDir + '*/*jpg'))
		self.imgCounts = {'train': trainCounts, 'valid': validCounts}

		# =======================================
		# Assign the class weights
		# =======================================

		if class_weight:
			self.imgWeights = {}

			classes = [i.split('/')[-1] for i in glob.glob(trainDir + '*')]
			for label in classes:
				labelCounts = len(glob.glob(trainDir + '*/*jpg'))
				self.imgWeights[label] = self.imgCounts['train'] / float(labelCounts)
		else:
			self.imgWeights = None	

		# =======================================
		# Get the image size and determine
		# the channel location
		# =======================================
		try:
			self.__inputShape
		except:
			from scipy.ndimage import imread
			self.__inputShape = imread(glob.glob(trainDir + '*/*jpg')[0]).shape
			self.__channel = 1 if np.argmin(self.__inputShape) == 0 else -1

	def _fitTrain(self, epochs):
		self.model.fit_generator(
			generator = self.imgTrain,
			validation_data = self.imgValid)

if __name__ == '__main__':

	cnn = hubble()
	cnn._setTrain(target='../img/GZ1/', class_weight=True)
