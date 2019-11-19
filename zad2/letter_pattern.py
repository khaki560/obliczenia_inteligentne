import math

def convert_to_vector(letter):
	vector = []
	for l in letter:
		vector.extend(l)
	return vector

def calculate_lenght(vector):
	return math.sqrt(sum([
		v*v for v in vector
	]))

def normalize(vector):
	length = calculate_lenght(vector)
	return [
		v/length for v in vector
	]
