import heapq
import math
import random
import sys
from xml.dom.minidom import Document,Node

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
                #if the exponent is too big, then just set the value to the system max float value
                #self[key] = sys.float_info[0]
                self.__items = [(element, math.exp(rationality*V)) for element, V in args.items()]
                self.normalize()
        elif isinstance(args, list):
            if rationality is None:
                self.__items = args
            else:
                self.__items = [(element, math.exp(rationality*V)) for element, V in args]
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

    def items(self):
        return iter(self.__items)


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
        return abs(self.probability()-1) < self.epsilon if epsilon is None else epsilon
        
    def __float__(self):
        return self.expectation()

    def sample(self):
        """
        :returns: an element from this domain, with a sample probability given by this distribution
        """
        return random.choices([item[0] for item in self.__items], [item[1] for item in self.__items])[0]

    def set(self, element):
        """
        Reduce distribution to be 100% for the given element
        :param element: the element that will be the only one with nonzero probability
        """
        self.clear()
        self[element] = 1

    def select(self, maximize=False):
        """
        Reduce distribution to a single element, sampled according to the given distribution
        :returns: the probability of the selection made
        """
        if maximize:
            element = self.max()
        else:
            element = self.sample()
        prob = self[element]
        self.set(element)
        return prob

    def max(self, k=1):
        """
        :param k: default is 1
        :returns: the top k most probable elements in this distribution (breaking ties by returning the highest-valued element)
        """
        if k == 1:
            element, prob = max(self.__items, key=lambda item: (item[1], item[0]))
            return element
        else:
            mass = 1
            heap = []
            for element, prob in self.__items:
                if len(heap) == k:
                    if prob > heap[0][0]:
                        mass += heap[0][0] - prob
                        heapq.heappop(heap)
                        heapq.heappush(heap, (prob, element))
                else:
                    mass -= prob
                    heapq.heappush(heap, (prob, element))
                if mass < heap[0][0]:
                    break
            return [tup[1] for tup in heap]

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
