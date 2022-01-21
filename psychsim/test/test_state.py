from psychsim.probability import *
from psychsim.pwl import *

import random

random.seed()

def gen_float(digits=1):
	return round(random.random()*pow(10, digits))/pow(10, digits)

def make_vector(size=5, name_start=ord('A')-1, constant=True):
	assert size > 0
	vector = KeyedVector()
	if constant:
		vector[CONSTANT] = gen_float()
	for i in range(size):
		var = chr(name_start+1+i)
		vector[var] = gen_float()
	print('Vector:', vector)
	return vector

def make_vector_distribution(num_elements=2, vector_size=5, name_start=ord('A')-1):
	elements = {make_vector(vector_size, name_start): gen_float() for i in range(num_elements)}
	dist = VectorDistribution(elements)
	dist.normalize()
	print('Distribution:', dist)
	return dist

def make_state(num_vars=5, max_uncertainty=2, num_splits=None):
	if num_splits is None:
		num_splits = num_vars-1
	splits = sorted(random.sample(list(range(num_vars-1)), num_splits))
	s = VectorDistributionSet()
	last = 0
	for split in splits:
		dist = make_vector_distribution(random.randint(1, max_uncertainty), split-last+1, ord('A')+last-1)
		s.add_distribution(dist)
		print(dist)
		last += split+1
	dist = make_vector_distribution(random.randint(1, max_uncertainty), num_vars-last, ord('A')+last-1)
	print(dist)
	s.add_distribution(dist)
	return s

def make_distribution(num_elements=10):
	dist = Distribution()
	while len(dist) < num_elements:
		dist.add_prob(gen_float(), gen_float(4))
	dist.normalize()
	return dist

def dont_test_max_size(max_size=3, num_iterations=10):
	for i in range(num_iterations):
		s = make_state(num_splits=1)
		print(s.keyMap)
		print(s.distributions)
		for sub, vec in s.distributions.items():
			print(vec.sortedString())
		print(s)

def test_top_k(num_iterations=10):
	for i in range(num_iterations):
		dist = make_distribution()
		assert abs(sum([tup[1] for tup in dist.items()]) - 1) < 1e-8
		for k in range(2, len(dist)):
			top = dist.max(k)
			assert len(top) == k
			floor = min([dist[element] for element in top])
			for element, prob in dist.items():
				if element not in top:
					assert prob < floor