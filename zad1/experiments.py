import random
from math import fabs
import neuron


def prepare_data(length, inputs_number, fun_to_calculate_result, data_range):
	'''
	Prepare random data used for neuron learing period

	Parameters
	----------
	length: int
		length of generating data
	inputs_number: int
		number of neuron inputs
	fun_to_calculate_result: function 
		function used to create correlation between inputs and output
	data_range: tupple, list
		min, max value use to generate data
	'''
	data = [[random.uniform(*data_range) for _ in range(inputs_number)] for _ in range(length)]
	result = [fun_to_calculate_result(a) for a in data]

	return data, result


def experiment(no, seed, dataset_learn_length, dataset_test_length, N, K, eta, w, p, fun_to_calculate_result):
	random.seed(a=150)

	learn_data, learn_result = prepare_data(
		length=dataset_learn_length,
		inputs_number=N,
		fun_to_calculate_result=fun_to_calculate_result,
		data_range=p
	)

	test_data, test_result = prepare_data(
		length=dataset_test_length,
		inputs_number=N,
		fun_to_calculate_result=fun_to_calculate_result,
		data_range=p
	)

	weight = [random.uniform(*w) for _ in range(N)]
	ne = neuron.Neuron(N, weight=weight)
	ne.train(learning_step=eta, epchos=K, inputs=learn_data, results=learn_result)

	error = [fabs(b-ne.calc(a)) for a, b in zip(test_data, test_result)]

	with open(f"wyniki\\Eksperyment_{no}.txt", 'w') as o_file:
		o_file.write(
			'\rwielkosc zbioru: {}\
			 \rLiczba Wejsc: {}\
			 \rLiczba Epok: {}\
			 \rkrok nauki: {}\
			 \rzakres poczatwkoch wartosci wag: {}\
			 \rzakres wartosci wejsciowych: {}\n\
			'.format(dataset_learn_length, N, K, eta, w, p)
		)
		o_file.write("\rsredni blad wgledny wynosi: {}\n".format(sum(error) / len(error)))
		o_file.write("\rWyniki testow:\n")
		for a, b in zip(test_data, test_result):
			o_file.write("\r{} == {}".format(b, ne.calc(a)))

	print(ne.delta_history[-1])
	ne.draw_delta(f'wyniki\\blad_delta_{no}.png')
	ne.draw_weights(f'wyniki\\wagi_{no}.png')


def main():
	experiment(
		no=1,
		seed=150,
		dataset_learn_length = 1,
		dataset_test_length = 20,
		N = 1,
		K = 50,
		eta = 0.0000009,
		w = [1.0, 2.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 2* a[0],
	)

	experiment(
		no=2,
		seed=150,
		dataset_learn_length = 1,
		dataset_test_length = 20,
		N = 2,
		K = 50,
		eta = 0.0000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : a[0] + a[1],
	)

	experiment(
		no=3,
		seed=150,
		dataset_learn_length = 1,
		dataset_test_length = 20,
		N = 2,
		K = 50,
		eta = 0.0000009,
		w = [0.99, 1.01],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : a[0] + a[1],
	)

	experiment(
		no=4,
		seed=150,
		dataset_learn_length = 10,
		dataset_test_length = 20,
		N = 3,
		K = 10,
		eta = 0.0000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)

	experiment(
		no=5,
		seed=150,
		dataset_learn_length = 20,
		dataset_test_length = 20,
		N = 3,
		K = 5,
		eta = 0.0000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)

	experiment(
		no=6,
		seed=150,
		dataset_learn_length = 5,
		dataset_test_length = 20,
		N = 3,
		K = 20,
		eta = 0.0000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)


	# Wiecej epok, zeby byla ladnie nauczona 
	experiment(
		no=7,
		seed=150,
		dataset_learn_length = 20,
		dataset_test_length = 20,
		N = 3,
		K = 20,
		eta = 0.0000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)


	experiment(
		no=8,
		seed=150,
		dataset_learn_length = 20,
		dataset_test_length = 20,
		N = 3,
		K = 20,
		eta = 0.0009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)

	experiment(
		no=9,
		seed=150,
		dataset_learn_length = 1000,
		dataset_test_length = 20,
		N = 3,
		K = 500,
		eta = 0.0000000009,
		w = [0.1, 10.0],
		p = [0, 1000],
		fun_to_calculate_result=lambda a : 6 * a[0] +  2 * a[1] - 3 * a[2],
	)

if __name__ == "__main__":
	main()
