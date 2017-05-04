import glob, sys
import numpy as np


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger

class hubble:
	
	def __init__(self, batch_size=32):
		self.__batchSize = batch_size


	def _setTrain(self, target, class_weight=False):
		"""
		Function for preparing the training and
		validation testsets.

		To call:
			_setTrain(target, weights=False)

		Parameters:
			target			target directory
			class_weight	(boolean) apply class weighting

		Postcondition:
			The ImageDataGenerators have been stored
			in the variables imgTrain and imgValid.

			The image counts for the training and
			validation sets have been stored in the
			dictionary imgCounts.

			The class weights have been stored in the
			variable imgWeights (if applicable).
		"""
		# =====================================================================
		# Set the train and valid directories
		# =====================================================================
		if target[-1] == '/':
			trainDir = target + 'train/'
			validDir = target + 'valid/'
		else:
			trainDir = target + '/train/'
			validDir = target + '/valid/'


		# =====================================================================
		# Create the image generators
		# =====================================================================
		datagen = ImageDataGenerator()

		self.imgTrain = datagen.flow_from_directory(
			directory = trainDir,
			batch_size = self.__batchSize,
			class_mode = 'categorical')

		self.imgValid = datagen.flow_from_directory(
			directory = validDir,
			batch_size = self.__batchSize,
			class_mode = 'categorical')

		# =====================================================================
		# Count the number of images
		# =====================================================================
		trainCounts = len(glob.glob(trainDir + '*/*jpg'))
		validCounts = len(glob.glob(validDir + '*/*jpg'))
		self.imgCounts = {'train': trainCounts, 'valid': validCounts}

		# =====================================================================
		# Assign the class weights
		# =====================================================================
		classes = [i.split('/')[-1] for i in glob.glob(trainDir + '*')]
		self.__classes = classes

		if class_weight:
			self.imgWeights = {}
			for label in classes:
				labelCounts = len(glob.glob(trainDir + label + '/*jpg'))
				self.imgWeights[label] = self.imgCounts['train'] / float(labelCounts)
		else:
			self.imgWeights = None	

		# =====================================================================
		# Get the image size and determine
		# the channel location
		# =====================================================================
		#try:
		#	self.__inputShape
		#except:
		#	from scipy.ndimage import imread
		#	self.__inputShape = imread(glob.glob(trainDir + '*/*jpg')[0]).shape
		#	self.__channel = 1 if np.argmin(self.__inputShape) == 0 else -1



	def _loadModel(self, modelfile):
		self.model = load_model(modelfile)


	def _loadWeights(self, weightfile):
		"""
		Function for loading weights

		To call:
			_loadWeights(weightfile)

		Parameters:
			weightfile		weight file to load
		"""
		try:
			self.model.load_weights(weights)
		except:
			print('Failed to load weights')
			sys.exit(1)


	def _createModel(self, neurons=1024, outs=None, act='elu', opt='rmsprop', drop=0.5, freeze=True, hidden=1):
		"""
		Function for adjusting the InceptionV3 model to have "outs" outputs.
		If _setTrain has been run, then leave "outs=None".

		To call:
			_createModel(neurons, outs, act, opt, drop, freeze)

		Parameters:
			neurons		number of neurons in dense layer
			outs		number of output layers (optional if _setTrain used)
			act			activation function
			opt			optimizer
			drop		dropout rate
			freeze		freeze the IV3 base layers
			hidden		number of hidden Dense/Drop layers to add
		"""
		if outs == None:
			try:
				outs = len(self.__classes)
			except:
				print('Please enter the number of outputs, or specify a training directory')
				sys.exit(1)

		# =====================================================================
		# Load the IV3 model, dropping the final layers
		# =====================================================================
		base_layer = InceptionV3(weights='imagenet', include_top=False)

		# =====================================================================
		# Add new layers for model training
		# =====================================================================
		mlayer = [] 
		mlayer.append( GlobalAveragePooling2d()(base_layer.output) )

		for _ in range(hidden):
			mlayer.append( Dense(neurons, activation=act)(mlayer[-1]) )
			mlayer.append( Drop(drop)(mlayer[-1]) )

		mlayer.append( Dense(outs, activation='softmax')(mlayer[-1]) )

		#pool_layer = GlobalAveragePooling2d()(base_layer.output)
		#dens_layer = Dense(neurons, activation=act)(pool_layer)
		#drop_layer = Drop(drop)(dens_layer)
		#pred_layer = Dense(outs, activation='softmax')(drop_layer)
		# model = Model(input=base_layer.input, output=pred_layer)

		# =====================================================================
		# Create the new model
		# =====================================================================
		model = Model(input=base_layer.input, output=mlayer[-1])

		# =====================================================================
		# Freeze the original weights ?
		# =====================================================================
		if freeze:
			for layer in base_layer.layers:
				layer.trainable = False

		# =====================================================================
		# Compile the model
		# =====================================================================
		model.compile(optimizer=opt, loss='categorical_crossentropy')

		self.model = model


	def _trainModel(self, epochs=100, logfile='train.log', savefile='weights.h5', save_weights_only=True, period=1):
		"""
		Function for training a model.

		To call:
			_trainModel(epochs, logfile, savefile, save_weights_only, period)

		Parameters:
			epochs				number of iterations
			logfile				file to save log-data to
			savefile			name of file to save weights/model
			save_weights_only	save only weights (or model)
			period				model check every "period" epochs
		"""

		# =====================================================================
		# Verify a model has been created
		# =====================================================================		
		try:
			self.model
		except:
			print('No model found')
			sys.exit(1)

		# =====================================================================
		# Test to see if the generators have been created
		# =====================================================================
		try:
			self.imgTrain
			self.imgCounts
			self.imgValid
		except:
			print('Not able to find dataset generators')
			sys.exit(1)

		# =====================================================================
		# Create callbacks for saving the best weights/model and log file
		# =====================================================================
		logger = CSVLogger(log)
		weight = ModelCheckpoint(savefile, monitor='val_loss', verbose=1, save_best_only=True, 
						save_weights_only=save_weights_only, mode='auto', period=period)

		# =====================================================================
		# Train the model
		# =====================================================================
		self.model.fit_generator(
			generator = self.imgTrain,
			steps_per_epoch = self.imgCounts['train'] // self.__batchSize,
			epochs = epochs,
			validation_data = self.imgValid,
			validation_steps = self.imgCounts['valid'] // self.__batchSize,
			callbacks = [logger, weight]
		)

if __name__ == '__main__':

	cnn = hubble()
	cnn._setTrain(target='../img/GZ1/', class_weight=True)
	cnn._createModel()
	cnn._trainModel(epochs=1)
