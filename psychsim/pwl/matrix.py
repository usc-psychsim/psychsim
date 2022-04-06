import copy

from psychsim.pwl.vector import KeyedVector, VectorDistribution
from psychsim.pwl.keys import CONSTANT, makeFuture


class KeyedMatrix:
    """
    Keeps rows in order by key
    """
    def __init__(self, arg=None):
        self.__keys_in = None
        self.__keys_out = None
        if isinstance(arg, dict):
            self.__rows = sorted(arg.items())
        elif isinstance(arg, list):
            self.__rows = sorted(arg)
        else:
            self.__rows = []
        self.__string = None
        
    def items(self):
        return iter(self.__rows)

    def __deepcopy__(self, memo):
        result = self.__class__(copy.deepcopy(self.__rows, memo))
        result.__keys_in = None  # self.__keys_in
        result.__keys_out = None  # self.__keys_out
        result.__string = None  # self.__string
        return result

    def __len__(self):
        return len(self.__rows)

    def __eq__(self, other):
        my_index = 0
        while my_index < len(self):
            my_key, my_vector = self.__rows[my_index]
            yr_index = 0
            while yr_index < len(other):
                yr_key, yr_vector = other.__rows[yr_index]
                if yr_key == my_key:
                    if my_vector != yr_vector:
                        return False
                    else:
                        break
                elif yr_key > my_key:
                    # Already passed where this key would be
                    return False
                else:
                    yr_index += 1
            else:
                # Did not find any vector for this key
                return False
            my_index += 1
        else:
            return True

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        result = self.__class__([(key, -vector) for key, vector in self.__rows])
        result.__keys_in = self.__keys_in
        result.__keys_out = self.__keys_out
        return result

    def __add__(self, other):
        items = []
        my_index = 0
        yr_index = 0
        while my_index < len(self):
            my_key, my_vector = self.__rows[my_index]
            if yr_index < len(other):
                yr_key, yr_vector = other.__rows[yr_index]
                if my_key == yr_key:
                    items.append((my_key, my_vector+yr_vector))
                    my_index += 1
                    yr_index += 1
                elif my_key < yr_key:
                    items.append((my_key, my_vector))
                    my_index += 1
                else: # my_key > yr_key
                    items.append((yr_key, yr_vector))
                    yr_index += 1
            else:
                # No more rows in other
                items.append((my_key, my_vector))
                my_index += 1
        # No more rows in me
        while yr_index < len(other):
            items.append(other.__rows[yr_index])
            yr_index += 1
        return KeyedMatrix(items)

    def __sub__(self, other):
        return self + (-other)

    def multiply_matrix(self, other):
        result = KeyedMatrix()
        result.__keys_out = self.getKeysOut()
        result.__keys_in = set()
        for r1, v1 in self.items():
            row = {}
            for c1, value1 in v1.items():
                try:
                    col = other[c1].items()
                except KeyError:
                    if c1 == CONSTANT:
                        col = [(CONSTANT, 1)]
                    else:
                        continue
                for c2, value2 in col:
                    row[c2] = row.get(c2, 0) + value1*value2
                    result.__keys_in.add(c2)
            result.__rows.append((r1, KeyedVector(row)))
        return result

    def multiply_vector(self, other):
        result = KeyedVector()
        for r1, v1 in self.items():
            for c1, value1 in v1.items():
                if c1 in other:
                    result[r1] = result.get(r1, 0) + value1*other[c1]
        return result

    def multiply_distribution(self, other):
        result = other.__class__([(self*vector, prob) for vector, prob in other.items()])
        result.remove_duplicates()
        return result

    def __mul__(self, other):
        """
        @warning: Muy destructivo for L{VectorDistributionSet}
        """
        if isinstance(other, KeyedMatrix):
            return self.multiply_matrix(other)
        elif isinstance(other, KeyedVector):
            return self.multiply_vector(other)
        elif isinstance(other, VectorDistribution):
            return self.multiply_distribution(other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, KeyedVector):
            # Transform vector
            other_items = sorted(other.items())
            result = {}
            my_index = 0
            yr_index = 0
            while my_index < len(self.__rows):
                my_key, my_vector = self.__rows[my_index]
                while yr_index < len(other_items):
                    yr_key, yr_value = other_items[yr_index]
                    if my_key == yr_key:
                        for col, my_value in my_vector.items():
                            result[col] = result.get(col, 0) + yr_value*my_value
                        my_index += 1
                        yr_index += 1
                        break
                    elif yr_key < my_key:
                        result[yr_key] = yr_value
                        yr_index += 1
                    else:
                        my_index += 1
                        break
                else:
                    break
            while yr_index < len(other_items):
                yr_key, yr_value = other_items[yr_index]
                result[yr_key] = yr_value
                yr_index += 1
            return other.__class__(result)
        elif isinstance(other, float) or isinstance(other, int):
            return self.__class__([(key, other*vector) for key, vector in self.items()])
        else:
            raise NotImplementedError
        return result
            
    def getKeysIn(self):
        """
        :returns: a set of keys which affect the result of multiplying by this matrix
        """
        if self.__keys_in is None:
            self.__keys_in = set().union(*[set(item[1].keys()) for item in self.__rows])
        return self.__keys_in

    def getKeysOut(self):
        """
        :returns: a set of keys which are changed as a result of multiplying by this matrix
        """
        if self.__keys_out is None:
            self.__keys_out = {item[0] for item in self.__rows}
        return self.__keys_out

    def keys(self):
        return self.getKeysIn() | self.getKeysOut()

    def desymbolize(self, table, debug=False):
        return self.__class__([(key, row.desymbolize(table)) for key, row in self.items()])

    def makeFuture(self, keyList=None):
        """
        Transforms matrix so that each row refers to only future keys
        :param keyList: If present, only references to these keys (within each row) are made future
        """
        return self.changeTense(True, keyList)

    def makePresent(self, keyList=None):
        return self.changeTense(False, keyList)
    
    def changeTense(self, future=True, keyList=None):
        """
        Transforms matrix so that each row refers to only future keys
        :param keyList: If present, only references to these keys (within each row) are made future
        """
        self.__string = None
        self.__keys_in = None
        for key, vector in self.items():
            vector.changeTense(future, keyList)
            
    def scale(self, table):
        result = self.__class__()
        for row, vector in self.items():
            if row in table:
                result[row] = KeyedVector()
                lo, hi = table[row]
                constant = 0
                for col, value in table.items():
                    if col == row:
                        # Same value
                        result[row][col] = value
                        constant += value*lo
                    elif col != CONSTANT:
                        # Scale weight for another feature
                        if abs(value) > epsilon:
                            assert col in table, 'Unable to mix symbolic and numeric values in single vector'
                            colLo, colHi = table[col]
                            result[row][col] = value*(colHi-colLo)*(hi-lo)
                            constant += value*colLo
                result[row][CONSTANT] = constant - lo
                if CONSTANT in vector:
                    result[row][CONSTANT] += vector[CONSTANT]
                result[row][CONSTANT] /= (hi-lo)
            else:
                result[row] = KeyedVector(vector)
        return result
        
    def __getitem__(self, key):
        for index, item in enumerate(self.__rows):
            if item[0] == key:
                return item[1]
            elif item[0] > key:
                # Past where it would be
                break
        raise KeyError(f'Matrix missing row for {key}')

    def __setitem__(self, key, value):
        assert isinstance(value, KeyedVector), 'Illegal row type: %s' % \
            (value.__class__.__name__)
        for index, item in enumerate(self.__rows):
            if item[0] == key:
                self.__rows[index] = (item[0], value)
                break
            elif item[0] > key:
                # Insertion sort
                self.__rows.insert(index, (key, value))
                break
        else:
            self.__rows.append((key, value))
        self._string = None

    def update(self, other):
        self.__string = None
        my_index = 0
        yr_index = 0
        while my_index < len(self):
            my_key, my_vector = self.__rows[my_index]
            while yr_index < len(other):
                yr_key, yr_vector = other.__rows[yr_index]
                if my_key == yr_key:
                    self.__rows[my_index] = (my_key, yr_vector)
                    my_index += 1
                    yr_index += 1
                    break
                elif my_key < yr_key:
                    my_index += 1
                    break
                else:  # my_key > yr_key
                    self.__rows.insert(my_index, (yr_key, yr_vector))
                    yr_index += 1
            else:
                break
        self.__rows += other.__rows[yr_index:]

    def __str__(self):
        if self.__string is None:
            self.__string = '\n'.join([f'{col}) {vec.hyperString()}' for col, vec in self.items()])
        return self.__string

    def __hash__(self):
        return hash(tuple(self.items()))


def dynamicsMatrix(key, vector):
    """
    :returns: a dynamics matrix setting the given key to be equal to the given weighted sum
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector(vector)})


def scaleMatrix(key, weight):
    """
    :returns: a dynamics matrix modifying the given keyed value by scaling it by the given weight
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({key: weight})})


def noChangeMatrix(key):
    """
    :returns: a dynamics matrix indicating no change to the given keyed value
    :rtype: L{KeyedMatrix}
    """
    return scaleMatrix(key, 1)


def nullMatrix(key):
    """
    :returns: a fake dynamics matrix that doesn't change time
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({key: KeyedVector({key: 1})})


def approachMatrix(key, weight, limit, limitKey=CONSTANT):
    """
    :param weight: the percentage by which you want the feature to approach the limit
    :type weight: float
    :param limit: the value you want the feature to approach
    :type limit: float
    :returns: a dynamics matrix modifying the given keyed value by approaching the given limit by the given weighted percentage of distance
    :rtype: L{KeyedMatrix}
    :param limitKey: the feature whose value to approach (default is CONSTANT)
    :type limitKey: str
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({key: 1-weight,
                                                      limitKey: weight*limit})})


def incrementMatrix(key, delta):
    """
    :param delta: the constant value to add to the state feature
    :type delta: float
    :returns: a dynamics matrix incrementing the given keyed value by the constant delta
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({key: 1, CONSTANT: delta})})


def setToConstantMatrix(key, value):
    """
    :type value: float
    :returns: a dynamics matrix setting the given keyed value to the constant value
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({CONSTANT: value})})


def setToFeatureMatrix(key, otherKey, pct=1, shift=0):
    """
    :type otherKey: str
    :returns: a dynamics matrix setting the given keyed value to a percentage of another keyed value plus a constant shift (default is 100% with shift of 0)
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({otherKey: pct,CONSTANT: shift})})


def addFeatureMatrix(key, otherKey, pct=1):
    """
    :type otherKey: str
    :returns: a dynamics matrix adding a percentage of another feature value to the given feature value (default percentage is 100%)
    :rtype: L{KeyedMatrix}
    """
    return KeyedMatrix({makeFuture(key): KeyedVector({key: 1, otherKey: pct})})


def setTrueMatrix(key):
    return setToConstantMatrix(key, 1)


def setFalseMatrix(key):
    return setToConstantMatrix(key, 0)
