from psychsim.probability import *
from psychsim.pwl import *

import copy
import random

random.seed()

def gen_float(digits=1):
	return round(random.random()*pow(10, digits))/pow(10, digits)

def make_vector(size=5, name_start=ord('A')-1, constant=True):
	assert size > 0
	vector = KeyedVector()
	if constant:
		vector[CONSTANT] = 1
	for i in range(size):
		var = chr(name_start+1+i)
		vector[var] = gen_float()
	return vector

def make_vector_distribution(num_elements=2, vector_size=5, name_start=ord('A')-1):
	assert vector_size > 0
	elements = {make_vector(vector_size, name_start): gen_float()+0.1 for i in range(num_elements)}
	dist = VectorDistribution(elements)
	dist.normalize()
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
		last += split+1
	dist = make_vector_distribution(random.randint(1, max_uncertainty), num_vars-last, ord('A')+last-1)
	s.add_distribution(dist)
	return s

def make_distribution(num_elements=10):
	dist = Distribution()
	while len(dist) < num_elements:
		dist.add_prob(gen_float(), gen_float(4))
	dist.normalize()
	return dist

def test_max_size(max_size=3, num_iterations=10):
	for i in range(num_iterations):
		s = make_state(num_splits=1, max_uncertainty=8)
		print(s)
		old = sorted([(world, prob) for world, prob in s.worlds()], key=lambda tup: tup[1], reverse=True)
		for world, prob in old:
			print(f'{prob} {world.items()}')
		s.prune_size(max_size)
		assert len(s) <= max_size
		print(s)
		for world, prob in sorted([(world, prob) for world, prob in s.worlds()], key=lambda tup: tup[1], reverse=True):
			print(f'{prob} {world.items()}')
		now = Distribution([(world, prob) for world, prob in s.worlds()])
		floor = min(now.values())
		for world, prob in old:
			if world in now.domain():
				assert prob >= floor
			else:
				assert prob <= floor

def test_top_k(num_iterations=10):
	for i in range(num_iterations):
		dist = make_distribution()
		assert abs(sum([tup[1] for tup in dist.items()]) - 1) < 1e-8
		top = dist.max()
		for element, prob in dist.items():
			if element != top:
				assert prob <= dist[top]
		for k in range(2, len(dist)):
			top = dist.max(k)
			assert len(top) == k
			floor = min([dist[element] for element in top])
			for element, prob in dist.items():
				if element not in top:
					assert prob < floor

def test_prune_k(num_iterations=10, num_elements=10):
	for i in range(num_iterations):
		for k in range(2, num_elements):
			dist = make_distribution(num_elements)
			original = list(dist.items())
			total = sum([dist[el] for el in dist.max(k)])
			prob = dist.prune_size(k)
			assert len(dist) == k
			assert prob == total
			floor = min([p for el, p in dist.items()])
			for item, prob in original:
				if item in dist.domain():
					assert prob >= floor
				else:
					assert prob <= floor
