from psychsim.pwl import *
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