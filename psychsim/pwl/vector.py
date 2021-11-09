import collections.abc
from xml.dom.minidom import Document,Node

from psychsim.probability import Distribution
from . import keys

class KeyedVector(collections.abc.MutableMapping):
    """
    Class for a compact, string-indexable vector
    :cvar epsilon: the margin used for equality of vectors (as well as for testing hyperplanes in L{KeyedPlane})
    :type epsilon: float
    :ivar _string: the C{str} representation of this vector
    :type _string: bool
    """
    epsilon = 1e-8

    def __init__(self,arg={}):
        collections.abc.MutableMapping.__init__(self)
        self._data = {}
        self._string = None
        if isinstance(arg,Node):
            self.parse(arg)
        elif arg:
            self._data.update(arg)

    def prune(self):
        for key,value in list(self.items()):
            if abs(value) < self.epsilon:
                del self[key]
        return self

    def __contains__(self,key):
        return key in self._data
    
    def __eq__(self,other):
        delta = 0.
        tested = {}
        for key,value in self.items():
            try:
                delta += abs(value-other[key])
            except KeyError:
                delta += abs(value)
            tested[key] = True
        for key,value in other.items():
            if key not in tested:
                delta += abs(value)
        return delta < self.epsilon

    def __ne__(self,other):
        return not self == other

    def __add__(self,other):
        result = KeyedVector(self)
        for key,value in other.items():
            result[key] = value + result.get(key,0.)
        return result

    def __neg__(self):
        result = KeyedVector()
        for key,value in self.items():
            result[key] = -value
        return result

    def __sub__(self,other):
        return self + (-other)

    def __mul__(self,other):
        if isinstance(other,KeyedVector):
            # Dot product
            total = 0
            if len(self) > len(other):
                for key,value in other.items():
                    if key in self:
                        total += value*self[key]
            else:
                for key,value in self.items():
                    if key in other:
                        total += value*other[key]
            return total
        elif isinstance(other,float):
            # Scaling
            result = KeyedVector({key: value*other for key,value in self.items()})
            return result
        else:
            return other.__rmul__(self)
#            raise NotImplementedError

    def __rmul__(self,other):
        if isinstance(other,float) or isinstance(other,int):
            result = self.__class__({key: other*value for key,value in self.items()})
            return result
        else:
            raise NotImplementedError

    def __imul__(self,other):
        """
        :type other: KeyedMatrix or KeyedTree
        """
        try:
            for row,vector in other.items():
                self[row] = self*vector
        except AttributeError:
            matrix = other[self]
            if isinstance(matrix,Distribution):
                result = VectorDistribution()
                for mat in matrix.domain():
                    vector = self.__class__(self)
                    vector *= mat
                    result.addProb(vector,matrix[mat])
                return result
            else:
                for row,vector in matrix.items():
                    self[row] = self*vector
        return self

    def copy_value(self, old_key, new_key):
        """
        Modifies the state so that the distribution over the new key's values is identical to that of the old key
        """
        self[new_key] = self[old_key]
        
    def rollback(self,future=None):
        if future is None:
            future = [key for key in self if keys.isFuture(key)]
        for key in future:
            self[keys.makePresent(key)] = self[key]
            del self[key]
        
    def __getitem__(self,key):
        return self._data[key]
    
    def __setitem__(self,key,value):
        self._string = None
        self._data[key] = value

    def __delitem__(self,key):
        self._string = None
        del self._data[key]

    def __iter__(self):
        return self._data.__iter__()
    
    def __len__(self):
        return len(self._data)

    def normalize(self):
        """
        Multiplies all of the weights so that the smallest weight is 1 and the relative values are preserved
        :return: the multiplier used
        """
        alpha = 1/min([abs(v) for v in self._data.values()])
        if abs(alpha-1) > self.epsilon:
            self._string = None
            for key in self._data.keys():
                self._data[key] *= alpha
        return alpha
            
    def desymbolize(self,table,debug=False):
        result = self.__class__()
        for key,value in self.items():
            if isinstance(value,float) or isinstance(value,int):
                result[key] = value
            else:
                result[key] = table[value]
        return result

    def makeFuture(self,keyList=None):
        """
        Transforms this vector to refer to only future versions of its columns
        :param keyList: If present, only references to these keys are made future
        """
        return self.changeTense(True,keyList)
        
    def makePresent(self,keyList=None):
        return self.changeTense(False,keyList)

    def changeTense(self,future=True,keyList=None):
        if keyList is None:
            keyList = self.keys()
        for key in keyList:
            if key in self and not key == keys.CONSTANT:
                if future:
                    assert not keys.isFuture(key)
                value = self[key]
                del self[key]
                if future:
                    self[keys.makeFuture(key)] = value
                else:
                    self[keys.makePresent(key)] = value
        
    def filter(self,ignore):
        """
        :return: a copy of me applying the given lambda expression to the keys (if a list is provided, then any keys in that list are dropped out)
        :rtype: KeyedVector
        """
        if isinstance(ignore,list):
            test = lambda k: not k in ignore
        else:
            test = ignore
        result = self.__class__()
        for key in filter(test,self.keys()):
            result[key] = self[key]
        return result

    def nearestNeighbor(self,vectors):
        """
        :return: the vector in the given set that is closest to me
        :rtype: KeyedVector
        """
        bestVector = None
        bestValue = None
        for vector in vectors:
            d = self.distance(vector)
            if bestVector is None or d < bestValue:
                bestValue = d
                bestVector = vector
        return bestVector

    def distance(self,vector):
        """
        :return: the distance between the given vector and myself
        :rtype: float
        """
        d = 0.
        for key in self.keys():
            d += pow(self[key]-vector[key],2)
        return d

    def __gt__(self,other):
        return sorted(self.items()) > sorted(other.items())
        
    def __str__(self):
        if self._string is None:
            mykeys = list(self.keys())
            mykeys.sort()
            self._string = '\n'.join(['%s: %s' % (k,self[k]) for k in mykeys])
        return self._string

    def sortedString(self):
        maxLength = max([len(key) for key in self])+1
        return '\n'.join(['{:{width}} {}'.format(key+':',value,width=maxLength) for key,value in sorted(self.items())])

    def hyperString(self):
        """
        :return: a string representation of this vector treating it as a weighted sum
        """
        def term2str(coef,key):
            if isinstance(coef,float):
                if key == keys.CONSTANT:
                    return '%5.3f' % (coef)
                else:
                    return '%5.3f*%s' % (coef,key)
            elif key == keys.CONSTANT:
                if isinstance(coef,int):
                    return '%d' % (coef)
                else:
                    return '%s' % (coef)
            elif coef == 1:
                return '%s' % (key)
            elif coef == -1:
                return '-%s' % (key)
            else:
                return '%d*%s' % (coef,key)
        rowStr = None
        items = sorted(self.items())
        if items[0][0] == keys.CONSTANT:
            items.append(items[0])
            del items[0]
        for key,coef in items:
            substr = term2str(coef,key)
            if rowStr is None:
                rowStr = substr
            elif substr[0] == '-':
                rowStr += substr
            else:
                rowStr += '+%s' % (substr)
        return rowStr

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,dict(self))

    def __hash__(self):
        return hash(tuple(self._data.items()))
#        return hash(frozenset(self._data.items()))

    def diff(self, other):
        my_keys = set(self.keys())
        yr_keys = set(other.keys())
        return sorted((my_keys-yr_keys)|(yr_keys-my_keys)|{k for k in my_keys&yr_keys if self[k] != other[k]})

class VectorDistribution(Distribution):
    """
    A class representing a L{Distribution} over L{KeyedVector} instances
    """

#    def __init__(self,args=None):
#        if args is None:
#            args = {KeyedVector({keys.CONSTANT:1}):1}
#        Distribution.__init__(self,args)

    def __contains__(self,key):
        return key in self.first()

    def keys(self):
        """
        :return: The keys of the vectors in the domain (assumed to be uniform),
        NOT the keys of the domain itself
        """
        if len(self) > 0:
            return {k for k in self.first().keys() if k != keys.CONSTANT}
        else:
            return set()
    
    def join(self,key,value):
        """
        Modifies the distribution over vectors to have the given value for the given key
        :param key: the key to the column to modify
        :type key: str
        :param value: either a single value to apply to all vectors, or else a L{Distribution} over possible values
        """
        original = dict(self)
        domain = self.domain()
        self.clear()
        for row in domain:
            prob = original[hash(row)]
            if isinstance(value,Distribution):
                for element in value.domain():
                    new = row.__class__(row)
                    new[key] = element
                    self.addProb(new,prob*value[element])
            else:
                row[key] = value
                self.addProb(row,prob)


    def merge(self,other,inPlace=False):
        """
        Merge two distributions (the passed-in distribution takes precedence over this one in case of conflict)
        :type other: VectorDistribution
        :param inPlace: if C{True}, modify this distribution directly; otherwise, return a new distribution (default is C{False})
        :type inPlace: bool
        :return: the merged distribution
        :rtype: VectorDistribution
        """
        if inPlace:
            result = self
        else:
            result = {}
        for old in self.domain():
            prob = self[old]
            del self[old]
            for diff in other.domain():
                new = old.__class__(old)
                new.update(diff)
                result.addProb(new, prob*other[diff])
        if inPlace:
            return self
        else:
            return self.__class__(result)
        
    def element2xml(self,value):
        return value.__xml__().documentElement

    def xml2element(self,key,node):
        return KeyedVector(node)

    def marginal(self,key):
        result = {}
        for row in self.domain():
            try:
                result[row[key]] += self[row]
            except KeyError:
                result[row[key]] = self[row]
        return Distribution(result)

    def select(self,maximize=False,incremental=False):
        """
        :param incremental: if C{True}, then select each key value in series (rather than picking out a joint vector all at once, default is C{False})
        """
        if incremental:
            # Sample each key and keep track how likely each individual choice was
            sample = KeyedVector()
            keys = self.domain()[0].keys()
            index = 0
            while len(self) > 1:
                key = keys[index]
                dist = self.marginal(key)
                if len(dist) > 1:
                    # Have to make a choice here
                    element,sample[key] = dist.sample(True)
                    # Figure out where the "spinner" ended up across entire pie chart
                    for other in dist.domain():
                        if other == element:
                            break
                        else:
                            sample[key] += dist[other]
                    for vector in self.domain():
                        if vector[key] != element:
                            del self[vector]
                    self.normalize()
                index += 1
            return sample
        else:
            return Distribution.select(self,maximize)
            
    def deleteKey(self,key):
        """
        Removes the specified column from all vectors in this distribution
        :type key: str
        """
        vectors = [(vec,self[vec]) for vec in self.domain()]
        self.clear()
        for vec,prob in vectors:
            del vec[key]
            self.addProb(vec,prob)
            
    def hasColumn(self,key):
        """
        :return: C{True} iff the given key appears in all of the vectors of this distribution
        :rtype: bool
        """
        for vector in self.domain():
            if not key in vector:
                return False
        return True

    def __rmul__(self,other):
        if isinstance(other,KeyedVector):
            result = {}
            for vector in self.domain():
                product = other*vector
                try:
                    result[product] += self[vector]
                except KeyError:
                    result[product] = self[vector]
            return Distribution(result)
        else:
            raise NotImplementedError

    def __imul__(self,other):
        original = [(vector, self[vector]) for vector in self.domain()]
        self.clear()
        for vector,prob in original:
            vector *= other
            if isinstance(vector,VectorDistribution):
                for vec in vector.domain():
                    self.addProb(vec,vector[vec]*prob)
            else:
                self.addProb(vector,prob)
        assert abs(sum([self[el] for el in self.domain()]) - 1)<1e-8,[self[el] for el in self.domain()]
        return self

    def prune(self,probThreshold,true=None):
        change = False
        for vec in self.domain():
            if self[vec] < probThreshold:
                if true:
                    for key in true:
                        if vec[key] != true[key]:
                            break
                    else:
                        # Has the true state, so don't delete
                        continue
                change = True
                del self[vec]
        if change:
            self.normalize()
                
    def __deepcopy__(self,memo):
        result = self.__class__({})
        for vector in self.domain():
            new = KeyedVector(vector)
            result[new] = self[vector]
        return result

    def copy_value(self, old_key, new_key):
        """
        Modifies the state so that the distribution over the new key's values is identical to that of the old key
        """
        original = dict(self)
        domain = self.domain()
        self.clear()
        for row in domain:
            assert isinstance(row, KeyedVector)
            prob = original[hash(row)]
            row[new_key] = row[old_key]
            self[row] = prob

    def rollback(self,future=None):
        original = [(vector, self[vector]) for vector in self.domain()]
        self.clear()
        for vector,prob in original:
            vector.rollback(future)
            self.addProb(vector,prob)
        return self

    def replace(self, substitution, key=None):
        """
        Replaces column values, either across all columns, or only for the specified column
        """
        for vector in self.domain():
            if key is None:
                key_list = vector.keys()
            elif not isinstance(key, list):
                key_list = [key]
            else:
                key_list = key
            for key in key_list:
                if vector[key] in substitution:
                    prob = self[vector]
                    del self[vector]
                    vector[key] = substitution[vector[key]]
                    self[vector] = prob

    def domain(self,key=None):
        if isinstance(key,str):
            return {v[key] for v in self.domain()}
        elif key is None:
            return super().domain()
        else:
            raise NotImplementedError('Domain available for only a single variable')

    def group_by_certainty(self, suppress_certain=False):
        vectors = list(self.domain())
        certain = {key: vectors[0][key] for key in self.keys()}
        for vector in vectors[1:]:
            for key, value in list(certain.items()):
                if vector[key] != value:
                    del certain[key]
        certain_str = str(self.__class__({vectors[0].__class__(certain): 1}))
        uncertain_str = str(self.__class__({vectors[0].__class__({key: vector[key] for key in self.keys()-certain.keys()}): self[vector]
            for vector in vectors}))
        return uncertain_str if suppress_certain else '{}\n{}'.format(certain_str, uncertain_str)
        
    def __str__(self):
        return '\n'.join(['%d%%\n%s' % (prob*100,vector.sortedString()) 
            for vector,prob in sorted(self.items(),key=lambda i: i[1],reverse=True)])