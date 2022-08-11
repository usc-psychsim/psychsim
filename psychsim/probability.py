import heapq
import math
import random
from typing import Any, Tuple


class Distribution:
    """
    A probability distribution over hashable objects
    """
    epsilon = 1e-8

    def __init__(self, args=None, rationality=None):
        """
        :param args: the initial elements of the probability distribution
        :type args: list/dict
        :param rationality: if not ``None``, then use as a rationality parameter in a quantal response (or inverse of temperature in a Bernoulli distribution) over the provided values
        :type rationality: float
        """
        if isinstance(args, dict):
            if rationality is None:
                self.__items = list(args.items())
            else:
                # subtract values by max value to avoid overflow
                max_V = max(args.values())
                self.__items = [(element, math.exp(rationality*(V-max_V))) for element, V in args.items()]
                self.normalize()
        elif isinstance(args, list):
            if rationality is None:
                self.__items = args
            else:
                # subtract values by max value to avoid overflow
                max_V = max(args)
                self.__items = [(element, math.exp(rationality*(V-max_V))) for element, V in args]
                self.normalize()
        else:
            self.__items = []

    def first(self):
        """
        :returns: the first element in this distribution's domain (most useful if there's only one element)
        """
        return self.__items[0][0]

    def get(self, element):
        try:
            return self[element]
        except ValueError:
            return 0

    def __contains__(self, element):
        return element in self.domain()

    def __getitem__(self, element):
        for other, prob in self.__items:
            if other == element:
                return prob
        else:
            raise ValueError(f'Element {element} not in domain')

    def __setitem__(self, element, probability):
        """
        :param element: the domain element
        :param probability: the probability to associate with the given key
        :type probability: float
        :warning: clobbers any previously existing probability value for this element
        """
        for index, content in enumerate(self.__items):
            if content[0] == element:
                self.__items[index] = (element, probability)
                break
        else:
            self.__items.append((element, probability))

    def __eq__(self, other):
        return set(self.__items) == set(other.__items)

    def items(self):
        return iter(self.__items)

    def values(self):
        return (item[1] for item in self.__items)

    def keys(self):
        return self.domain()

    def add_prob(self, element, probability):
        """
        Utility method that increases the probability of the given element by the given value
        :param element: the domain element
        :param probability: the probability to add for this element
        :type probability: float
        """
        for index, content in enumerate(self.__items):
            if content[0] == element:
                self.__items[index] = (element, content[1]+probability)
                break
        else:
            self.__items.append((element, probability))

    def addProb(self, element, value):
        """
        Utility method that increases the probability of the given element by the given value
        """
        return self.add_prob(element, value)

    def getProb(self,element):
        raise DeprecationWarning('Use get method instead')

    def __delitem__(self, element):
        for index, content in enumerate(self.__items):
            if content[0] == element:
                del self.__items[index]
                break
        else:
            raise ValueError(f'Element {element} not in domain')

    def clear(self):
        self.__items.clear()

    def replace(self, old, new):
        """Replaces on element in the sample space with another.  Raises an exception if the original element does not exist, and an exception if the new element already exists (i.e., does not do a merge)
        """
        if new in [item[0] for item in self.__items]:
            raise ValueError(f'Element {new} already exists in this distribution')
        for index, content in enumerate(self.__items):
            if content[0] == old:
                self.__items[index] = (new, content[1])
                break
        else:
            raise ValueError(f'Element {old} not in domain')

    def domain(self):
        """
        :returns: the sample space of this probability distribution
        :rtype: generator
        """
        return (item[0] for item in self.__items)

    def normalize(self):
        """Normalizes the distribution so that the sum of values = 1
        """
        total = self.probability()
        try:
            self.__items = [(element, probability/total) for element, probability in self.__items]
        except ZeroDivisionError:
            raise ValueError('Cannot normalize a distribution with 0 probability mass')

    def __len__(self):
        return len(self.__items)

    def expectation(self):
        """
        :returns: the expected value of this distribution
        :rtype: float
        """
        if len(self) == 1:
            # Shortcut if no uncertainty
            return self.first()
        else:
            return sum([element*probability for element, probability in self.__items])

    def remove_duplicates(self):
        """
        Makes sure all elements are unique (combines probability mass when appropriate)
        :warning: modifies this distribution in place
        """
        i = 0
        while i < len(self.__items)-1:
            j = i+1
            while j < len(self.__items):
                if self.__items[i][0] == self.__items[j][0]:
                    self.__items[i] = (self.__items[i][0], self.__items[i][1]+self.__items[j][1])
                    del self.__items[j]
                else:
                    j += 1
            i += 1

    def probability(self):
        """
        :return: the total probability mass in this distribution
        """
        return sum([item[1] for item in self.__items])

    def is_complete(self, epsilon=None):
        """
        :return: True iff the total probability mass is 1 (or within epsilon of 1)
        """
        return math.isclose(self.probability(), 1, rel_tol=self.epsilon if epsilon is None else epsilon)
        
    def __float__(self):
        return self.expectation()

    def sample(self) -> Tuple[Any, float]:
        """
        :returns: an element from this domain, with a sample probability given by this distribution
        """
        return random.choices(self.__items, [item[1] for item in self.__items])[0]

    def set(self, element):
        """
        Reduce distribution to be 100% for the given element
        :param element: the element that will be the only one with nonzero probability
        """
        self.__items = [(element, 1)]

    def select(self, maximize=False):
        """
        Reduce distribution to a single element, sampled according to the given distribution
        :returns: the probability of the selection made
        """
        if maximize:
            element = self.max()
            prob = self[element]
        else:
            element, prob = self.sample()
        self.set(element)
        return prob

    def max(self, k=1, number=1):
        """
        :param k: default is 1
        :param number of values to return for each element (element if 1, element and probability if 2, element probability and index in domain if 3)
        :returns: the top k most probable elements in this distribution (breaking ties by returning the highest-valued element)
        """
        mass = 1
        heap = []
        for index, tup in enumerate(self.__items):
            element, prob = tup
            if len(heap) == k:
                if prob > heap[0][0]:
                    mass += heap[0][0] - prob
                    heapq.heapreplace(heap, (prob, element, index))
            else:
                mass -= prob
                heapq.heappush(heap, (prob, element, index))
            if len(heap) == k and mass < heap[0][0]:
                break
        if number == 1:
            if k == 1:
                return heap[0][1]
            else:
                return [tup[1] for tup in heap]
        elif number == 2:
            if k == 1:
                return (heap[0][1], heap[0][0])
            else:
                return [(tup[1], tup[0]) for tup in heap]
        elif number == 3:
            if k == 1:
                return (heap[0][1], heap[0][0], heap[0][2])
            else:
                return heap

    def entropy(self):
        """
        :returns: entropy (in bits) of this distribution
        """
        return sum([-item[1]*math.log2(item[1]) for item in self.__items])

    def __add__(self, other):
        if isinstance(other, Distribution):
            result = self.__class__()
            for my_el, my_prob in self.__items:
                for yr_el, yr_prob in other.__items:
                    result.add_prob(my_el+yr_el, my_prob*yr_prob)
            return result
        else:
            return self.__class__([(element+other, prob) for element, prob in self.__items])

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self.__class__([(-element, prob) for element, prob in self.__items])

    def __mul__(self, other):
        if isinstance(other, Distribution):
            result = self.__class__()
            for my_el, my_prob in self.__items:
                for yr_el, yr_prob in other.__items:
                    result.add_prob(my_el*yr_el, my_prob*yr_prob)
            return result
        else:
            return self.__class__([(element*other, prob) for element, prob in self.__items])

    def scale_prob(self, factor):
        """
        :return: a new Distribution whose probability values have all been multiplied by the given factor
        """
        return self.__class__([(element, prob*factor) for element, prob in self.__items])

    def prune(self, epsilon=1e-8):
        raise DeprecationWarning('Use prune_probability, prune_elements, or prune_size instead')

    def prune_probability(self, threshold, normalize=False):
        """
        Removes any elements in the distribution whose probability is strictly less than the given threshold
        :param normalize: Normalize distribution after pruning if True (default is False)
        :return: the probability mass remaining after pruning (and before any normalization)
        """
        self.__items = [(element, prob) for element, prob in self.__items if prob >= threshold]
        prob = self.probability()
        if normalize:
            self.normalize()
        return prob

    def prune_size(self, k):
        """
        Remove least likely elements to get domain to size k
        :returns: the remaining total probability
        """
        mass = 1
        heap = []
        for i, tup in enumerate(self.__items):
            element, prob = tup
            if len(heap) == k:
                if prob > heap[0][0]:
                    mass += heap[0][0] - prob
                    heapq.heapreplace(heap, (prob, i))
            else:
                mass -= prob
                heapq.heappush(heap, (prob, i))
            if len(heap) == k and mass < heap[0][0]:
                break
        self.__items = [self.__items[tup[1]] for tup in heap]
        return self.probability()

    def prune_elements(self, epsilon=1e-8):
        """
        Merge any elements that are within epsilon of each other
        """
        i = 0
        while i < len(self)-1:
            el1, p1 = self.__items[i]
            j = i+1
            while j < len(self):
                el2, p2 = self.__items[j]
                if abs(el1-el2) < epsilon:
                    self.__items[i] = (el1, p1+p2)
                    del self.__items[j]
                else:
                    j += 1
            i += 1

    def sorted_string(self):
        return '\n'.join([f'{int(round(100*prob)): >3d}%: '+self.element_to_str(element).replace("\n", "\n\t") for element, prob in sorted(self.__items, key=lambda item: item[0])])

    def __str__(self):
        return '\n'.join([f'{int(round(100*prob)): >3d}%: '+self.element_to_str(element).replace("\n", "\n\t") for element, prob in self.__items])

    def element_to_str(self, element):
        return str(element)
        
#    def __hash__(self):
#        return hash(self.__items)

    def __copy__(self):
        return self.__class__(self.__items[:])

    def __getstate__(self):
        return self.__items

    def __setstate__(self, state):
        self.__items = state
