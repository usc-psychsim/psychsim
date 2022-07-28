from psychsim.pwl.keys import WORLD, CONSTANT, makeFuture
from psychsim.pwl.vector import KeyedVector
from psychsim.pwl.matrix import KeyedMatrix, setTrueMatrix, setFalseMatrix
from psychsim.pwl.plane import trueRow, KeyedPlane
from psychsim.pwl.tree import makeTree
from psychsim.world import World


def test_constant_comparison():
    world = World()
    p = world.defineState(WORLD, 'p', int)
    world.setFeature(p, 0)
    tree = {'if': KeyedPlane(KeyedVector({CONSTANT: 1}), 0, 1),
            True: KeyedMatrix({makeFuture(p): KeyedVector({CONSTANT: 3})}),
            False: KeyedMatrix({makeFuture(p): KeyedVector({CONSTANT: 1})})}
    world.setDynamics(p, True, makeTree(tree))

    a = world.addAgent('Agent')
    a.addAction({'verb': 'noop'})
    world.setOrder([{a.name}])

    world.step()    


def test_nochange():
    world = World()
    x = world.defineState(WORLD, 'x', bool)
    world.setFeature(x, False)
    y = world.defineState(WORLD, 'y', bool)
    world.setFeature(y, False)
    world.setDynamics(y, True, makeTree({'if': trueRow(makeFuture(x)), True: setTrueMatrix(y), False: setFalseMatrix(y)}))
    a = world.addAgent('Agent')
    a.addAction({'verb': 'noop'})
    world.setOrder([{a.name}])

    world.step()
