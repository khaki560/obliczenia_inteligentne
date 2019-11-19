

class NeuralNetwork(object):

	def __init__(self, layers):
		'''
		Parameters
		----------
		layers : List
			configuration of number of nuerons for each layer 
		'''
		self.layers = layers

	#TODO: NO TESTED
	def add_layer(self, new_layer):
		'''
		add new layer at the end of neutral network

		Parameters
		----------
		new_layer: list
			list of neurons, that will be part of this layer

		'''
		self.layers.append(new_layer)

	#TODO: NO TESTED
	def configure_layer(self, no, new_layer):
		''' 
		reconfigure single layer by providing new layer of neurons

		Parameters
		----------
		no : int 
			number of layer to reconfigure
		new_layer: list
			list of neurons, that will be part of this layer

		'''
		assert no > len(new_layer)
		self.layers[no] = new_layer

	def calc_single_layer(self, no, inputs):
		'''
		Calculate output for signle layer of neural network

		Parameters
		----------
		no : int 
			number of layer
		input : list 
			vector of inputs 
		
		Output: list
			Return vector of calculated output of every neuron of single layer
		'''
		return [
				neuron.calc(i)
				for i, neuron in zip(inputs, self.layers[no])
			]


	def calc(self, inputs):
		'''
		Calculate output of neural Network

		inputs : list
			1d list of inputs

		'''
		#Calculate first layer
		inputs = self.calc_single_layer(0, inputs)

		#Calulare rest of layers
		for layer_no in range(1, len(self.layers)):
			inputs = [inputs for _ in range(len(self.layers[layer_no]))]
			inputs = self.calc_single_layer(layer_no, inputs)

		return inputs


if __name__ == "__main__":
	import neuron
	from letter_pattern import *
	
	x_norm = normalize(convert_to_vector(X))
	y_norm = normalize(convert_to_vector(Y))
	z_norm = normalize(convert_to_vector(Z))

	nn = NeuralNetwork(
		[
			[neuron.Neuron(1, None, None, [1]) for _ in range(16)],
			[
				neuron.Neuron(1, None, None, x_norm),
				neuron.Neuron(1, None, None, y_norm),
				neuron.Neuron(1, None, None, z_norm),
			],
		])


	print(nn.calc(inputs=x_norm))
