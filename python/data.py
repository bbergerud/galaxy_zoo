import numpy as np
import glob, os, sys
from scipy import misc
from sklearn.utils import shuffle

class data:
	'''
	Class associated with loading the data and data
	transformations.

	Functions:
		_createDataArray(...)

		_loadDataArray(train=False, valid=False, test=False)
			Parameters are strings. Those that are not equal
			to false and loaded and stored in self.X_*, where 
			* represents the parameter name.

		_removeDataArray(...)
	'''

	def __init__(self, *args, **kwargs):
		pass

	def _createDataArray(self, datafolder, output, train=True, valid=True, test=True):
		'''
		Function for creating the train, valid, and test
		arrays.

		To call:
			_createDataArray(datafolder, output, train, valid, test)

		Parameters:
			datafolder		folder that contains the image
			output			folder to output the data to
		'''

		# =====================================================
		# Create a list of all the subdirectories in the
		# datafolder. Verify that at least one directory
		# was identified.
		# =====================================================
		folders = list()

		for root, dirs, files in os.walk(datafolder):
			if not dirs:
				folders.append(root)

		if folders == []:
			print("No folders located. Aborting")
			sys.exit(1)


		# =====================================================
		# Get the label and partition type (train, valid, test)
		# from the string path
		# =====================================================
		y_train = []
		y_valid = []
		y_test  = []

		for folder in folders:

			# =====================================================
			# Get the label and partition type (train, valid, test)
			# from the string path
			# =====================================================
			partition, label = folder.split('/')[-2:]

			if (partition == 'train') and not train:
				continue
			if (partition == 'valid') and not valid:
				continue
			if (partition == 'test') and not test:
				continue

			print("{:s}\t{:s}".format(partition, label))

			# =====================================================
			# Retrieve all the images in the folder and open
			# =====================================================
			files = glob.glob(folder + '/*jpg')

			imgs = []
			for fn in files:

				img = misc.imread(fn)
				imgs.append(img)

			imgs = np.asarray(imgs)
			nimg = imgs.shape[0]

			# =====================================================
			# Insert the images into the appropriate array
			# and store the labels
			# =====================================================
			if partition.lower() == 'valid':
				try:
					X_valid = np.concatenate((X_valid, imgs), axis=0)
				except:
					X_valid = imgs

				y_valid.extend(nimg * [label])

			elif partition.lower() == 'test':
				try:
					X_test = np.concatenate((X_test, imgs), axis=0)
				except:
					X_test = imgs

				y_test.extend(nimg * [label])

			else:
				try:
					X_train = np.concatenate((X_train, imgs), axis=0)
				except:
					X_train = imgs

				y_train.extend(nimg * [label])

		y_train = np.asarray(y_train)
		y_valid = np.asarray(y_valid)
		y_test  = np.asarray(y_test)


		# =====================================================
		# Check if the output directory exists. If not, make
		# =====================================================
		if not os.path.exists(output):
			try:
				os.makedirs(output)	
			except:
				print("Failed to create output directory.")
				print("Will store files in the current directory")
				output = './'

		# =====================================================
		# Save the files
		# =====================================================
		if output[-1] != '/':
			output += '/'


		try:
			# Check if the array exists
			X_train

			# Shuffle the arrays
			X_train, y_train = shuffle(X_train, y_train, random_state=1482)
			X_train, y_train = shuffle(X_train, y_train, random_state=2642)

			# Save the results
			np.save(output + "X_train.npy", X_train)
			np.save(output + "y_train.npy", y_train)

		except:
			print("No training dataset / Failure to save")



		try:
			# Check if the array exists
			X_valid

			# Shuffle the arrays
			X_valid, y_valid = shuffle(X_valid, y_valid, random_state=5232)
			X_valid, y_valid = shuffle(X_valid, y_valid, random_state=1523)

			# Save the results
			np.save(output + "X_valid.npy", X_valid)
			np.save(output + "y_valid.npy", y_valid)

		except:
			print("No validation dataset / Failure to save")


		try:
			# Check if the array exists
			X_test

			# Shuffle the arrays
			X_test, y_test = shuffle(X_test, y_test, random_state=9382)
			X_test, y_test = shuffle(X_test, y_test, random_state=3823)

			# Save the results
			np.save(output + "X_test.npy", X_test)
			np.save(output + "y_test.npy", y_test)

		except:
			print("No testing dataset / Failure to save")



	def _loadDataArray(self, datafolder, train=True, valid=True, test=True):
		'''
		Function for loading in a 4D tensor that represents
		the collection of images to classify. Used to load
		the train, validation, and test sets.

		To call:
			_loadDataArray(datafolder, train, valid, test)

		Parameters:
			datafolder		path to folder containing dataset
			train			[boolean] load training dataset
			valid			[boolean] load validation dataset
			test			[boolean] load testing dataset

		Postcondition:
			If the input parameter * is a string, then the
			4D tensor is loaded and stored in self.X_*
		'''
		if datafolder[-1] != '/':
			datafolder += '/'

		if train:
			try:
				self.X_train = np.load(datafolder + 'X_train.npy')
				self.y_train = np.load(datafolder + 'y_train.npy')
			except:
				print("Failed to load a train file")

		if valid: 
			try:
				self.X_valid = np.load(datafolder + 'X_valid.npy')
				self.y_valid = np.load(datafolder + 'y_valid.npy')
			except:				
				print("Failed to load a valid file")

		if test:  
			try:
				self.X_test = np.load(datafolder + 'X_test.npy')
				self.y_test = np.load(datafolder + 'y_test.npy')
			except:
				print("Failed to load a test file")




	def _removeDataArray(self, train=True, valid=True, test=True):
		'''
		Function for removing a dataset from memory.

		To call:
			_removeDataArray(train, valid, test)

		Parameters:
			train		[boolean] remove training dataset
			valid		[boolean] remove validation dataset
			test		[boolean] remove testing dataset
		'''
		if train:
			try:
				del self.X_train, self.y_train
			except:
				print("Failed to delete train")

		if valid:
			try:
				del self.X_valid, self.y_valid
			except:
				print("Failed to delete valid")

		if test:
			try:
				del self.X_test, self.y_test
			except:
				print("Failed to delete test")



if __name__ == '__main__':

	hubble = data()
	hubble._createDataArray('../img/GZ1/', '../img/GZ1/')
	hubble._loadDataArray('../img/GZ1/')
