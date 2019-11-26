import sys
sys.path.append('..')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import zad1.neuron as neuron
from letter_pattern import *
from neural_network import *





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

X = [
	[1, 0, 0, 1],
	[0, 1, 1, 0],
	[0, 1, 1, 0],
	[1, 0, 0, 1],
]

X1 = [
	[0, 0, 0, 0],
	[0, 1, 0, 1],
	[0, 0, 1, 0],
	[0, 1, 0, 1],
]



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

		
	# print(image[0])
	# print(image[1])
	# print(image[2])
	# print(image[3])
	# print(image[4])

	# lista = []
	# for x in image:
	# 	for y in x:
	# 		print(y)
	# 		print(sum(y))
				
	a =  [ 
		# [ (1 if sum(y) >= 4 else 1) for y in x] 
		[ sum(y) for y in x] 
		for x in image
	]

	# print(a)
	i = 0
	j = 0

	while i < len(a):
		j = 0
		while j < len(a[i]):
			if int(a[i][j]) > 255 :
				a[i][j] = 0
			else:
				a[i][j] = 1
			j += 1
		i += 1
	return a



def main():
	# a = load_image('images\\white.bmp')
	# a = load_image('images\\black.bmp')
	# print(a)
	# exit()
	
	a_norm = normalize(convert_to_vector(load_image('images\\a.bmp')))

	x_norm = normalize(convert_to_vector(load_image('images\\x.bmp')))
	
	# x_norm = normalize(convert_to_vector(X))
	y_norm = normalize(convert_to_vector(load_image('images\\y.bmp')))
	z_norm = normalize(convert_to_vector(load_image('images\\z.bmp')))

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
		# ["x.bmp",          X1],
		["x.bmp",          load_image('images\\x.bmp')],
		["y.bmp",          load_image('images\\y.bmp')],
		["z.bmp",          load_image('images\\z.bmp')],
		["x_popsute.bmp",  load_image("images\\x_popsute.bmp")],
		["y_popsute.bmp",  load_image("images\\y_popsute.bmp")], 
		["y_wiecej.bmp",   load_image("images\\y_wiecej.bmp")],  
		["y_trudne.bmp",   load_image("images\\y_trudne.bmp")],
		["y_trudne_2.bmp", load_image("images\\y_trudne_2.bmp")],
		["a.bmp",          load_image("images\\a.bmp")],
		["a_damage.bmp",   load_image("images\\a_damage.bmp")],
		["a_damage_2.bmp", load_image("images\\a_damage_2.bmp")],
		["black.bmp", load_image("images\\black.bmp")],
		["white.bmp", load_image("images\\white.bmp")], 
	]
	# print(test_images[0][1])
	# a = convert_to_vector(test_images[0][1])
	# normalize(a)
	# exit()
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
	# a = [0.35355339059327373, 0, 0, 0.35355339059327373, 0, 0.35355339059327373, 0.35355339059327373, 0, 0, 0.35355339059327373, 0.35355339059327373, 0, 0.35355339059327373, 0, 0, 0.35355339059327373]
	# b = [0, 0, 0, 0, 0, 0.4472135954999579, 0, 0.4472135954999579, 0, 0, 0.4472135954999579, 0, 0, 0.4472135954999579, 0, 0.4472135954999579]


	# s = 0
	# for _a, _b in zip(a,b):
	# 	s += _a * _b

	# print(s)
