import sys
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
sys.path.append('..')

import zad1.neuron as neuron
from neural_network import *
from letter_pattern import *


LETTERS = {
	-1 : "None",
	0  : "X",
	1  : "Y",
	2  : "Z",
	3  : "a",
}


def decide(out):
	'''
	Decide with letter was found

	Parameters
	----------
	out : List
		list of outputs of neural network
	Output
	----------
	str : letter that was detected
	'''
	maximum = -1
	iterator = -2

	for i, o in enumerate(out):
		if o > maximum:
			maximum = o
			iterator = i
	return LETTERS[iterator]


def load_image(path):
	'''
	Load image and convert it to array of zeros and ones

	Parameters
	----------
	path : str 
		path to image

	Output
	----------
	list: List full of zeros and ones

	'''
	img = mpimg.imread(path) 
	image = img.tolist()
		
	return [ 
		[ 0 if sum(y) > 0 else 1 for y in x] 
		for x in image
	]


def main():
	a_norm = normalize(convert_to_vector(load_image('images\\a.png')))
	x_norm = normalize(convert_to_vector(load_image('images\\x.png')))
	y_norm = normalize(convert_to_vector(load_image('images\\y.png')))
	z_norm = normalize(convert_to_vector(load_image('images\\z.png')))

	size_of_letter = len(x_norm)

	#Create neural Network
	nn = NeuralNetwork([
		#wartswa kopiujacą 
		[neuron.Neuron(1, [1]) for _ in range(size_of_letter)],
		#warstwa wyjściowa
		[
			neuron.Neuron(size_of_letter, x_norm),
			neuron.Neuron(size_of_letter, y_norm),
			neuron.Neuron(size_of_letter, z_norm),
			neuron.Neuron(size_of_letter, a_norm), 
		],
	])


	#tests

	test_images = [
		["x.png",          load_image('images\\x.png')],
		["y.png",          load_image('images\\y.png')],
		["z.png",          load_image('images\\z.png')],
		["x_popsute.png",  load_image("images\\x_popsute.png")],
		["y_popsute.png",  load_image("images\\y_popsute.png")], 
		["y_wiecej.png",   load_image("images\\y_wiecej.png")],  
		["y_trudne.png",   load_image("images\\y_trudne.png")],
		["y_trudne_2.png", load_image("images\\y_trudne_2.png")],
		["a.png",          load_image("images\\a.png")],
		["a_damage.png",   load_image("images\\a_damage.png")],
		["a_damage_2.png", load_image("images\\a_damage_2.png")],
	]

	test_normalize_vectors = [(array[0], normalize(convert_to_vector(array[1]))) for array in test_images ]

	for label, vector in test_normalize_vectors:
		results = nn.calc(vector)
		decision = decide(results)
		print("Podano", label)
		print("Neuron X odpowiedzial z", results[0])
		print("Neuron Y odpowiedzial z", results[1])
		print("Neuron Z odpowiedzial z", results[2])
		print("Neuron A odpowiedzial z", results[3])
		print("Wybrano ", decision)
		print('\n')



if __name__ == "__main__":
	main()
