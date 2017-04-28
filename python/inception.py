from keras.models import Model
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input


class inception:

	def __init__(self, batch_size=32, l2reg=0.0002, input_shape=(299,299,3)):
		self.__batchSize = batch_size
		self.__l2reg = l2reg
		self.__inputShape = input_Shape
		self.__channel = 1 if np.argmin(self.__inputShape) == 0 else -1


	def __testModel(self):
		"""
		Function for testing whether a model has been initialized.
		If not, the model instance is created.
		"""
		try:
			self.__model
		except:
			self.__model = []

	def _conv2D(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, 
			dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
			bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
			activity_regularizer=None, kernel_constraint=None, bias_constraint=None):


		self.__testModel()


	def _flatten(self):

		self.__model.append(Flatten()(self.model_[-1]))
