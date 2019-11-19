import random
import matplotlib.pyplot as plt

class Neuron(object):
	def __init__(self, inputs_number, weight):
		"""
		Parameters
		----------
		inputs_number: int
			number of Neuron inputs
		weight: list, tuple 
			initial weights of neuron
		"""
		if len(weight) != inputs_number:
			raise Exception("number of weights must be equal to number of inputs")

		self.inputs = inputs_number
		self.weight = weight

		self.weight_history = [ [] for _ in range(inputs_number)]
		self.delta_history = []

		self.to_file = []

	def calc(self, inputs):
		''' 
		Calculated single neuron output

		Parameters
		----------
		inputs: list
			array of values, provide to neuron inputs

		'''
		return sum([a * b for a, b in zip(inputs, self.weight)])

	def calc_delta(self, output, desired):
		'''
		calculate error absolute error between neurop output and excepted result

		Parameters
		----------
		output: float
			value calculated by neuron
		desired: float 
			excepted value
		'''
		return desired - output

	def __train(self, learning_step, inputs, results):
		'''
		'''
		for i, r in zip(inputs, results):
			y = self.calc(i)
			delta = self.calc_delta(output=y, desired=r)
			self.delta_history.append(delta)
			self.weight = [_w + learning_step * delta * _i for _w, _i in zip(self.weight, i)]
			for i in range(len(self.weight)):
				self.weight_history[i].append(self.weight[i])


	def train(self, learning_step, epchos, inputs, results):
			'''
			train single neuron with delta rule

			Parameters
			----------
			learning_step: float
				number, that represent speed of learning curve
			epchos: int
				number of iterations through all data set 
			inputs: list
				array of values, provide to neuron inputs
			result: list
				excepted values on neuron output
			'''
			del(self.delta_history[:])
			for i in range(len(self.weight_history)):
				del(self.weight_history[i][:])

			for _ in range(epchos):
				self.__train(learning_step, inputs, results)

	def draw_weights(self, path=None):
		'''
		draw weights
		'''
		plt.clf()
		for w in self.weight_history:
			plt.plot(w)
		plt.title('Wagi neuronu')
		plt.xlabel("Iteracja [-]")
		plt.ylabel("bląd iteracji [-]")
		if path is not None:
			plt.savefig(path)
		else:
			plt.show()

	def draw_delta(self, path=None):
		'''
		draw delta parameter
		'''
		plt.clf()
		plt.plot(self.delta_history)
		plt.title('Wspołczynnik Delta')
		plt.xlabel("Iteracja [-]")
		plt.ylabel("bląd iteracji [-]")
		if path is not None:
			plt.savefig(path)
		else:
			plt.show()