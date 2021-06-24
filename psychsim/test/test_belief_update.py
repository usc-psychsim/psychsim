from psychsim.action import *
from psychsim.world import *
from psychsim.agent import Agent
from psychsim.pwl import *
from psychsim.reward import *

def setup_world():
    # Create world
    world = World()
    # Create agents
    tom = world.addAgent('Tom')
    jerry = world.addAgent('Jerry')
    return world


def add_state(world):
    """Create state features"""
    world.defineState('Tom','health',int,lo=0,hi=100,
                           description='%s\'s wellbeing' % ('Tom'))
    world.setState('Tom','health',50)
    world.defineState('Jerry','health',int,lo=0,hi=100,
                           description='%s\'s wellbeing' % ('Jerry'))
    world.setState('Jerry','health',50)


def add_actions(world,order=None):
    """Create actions"""
    actions = {}
    actions['chase'] = world.agents['Tom'].addAction({'verb': 'chase','object': 'Jerry'})
    actions['hit'] = world.agents['Tom'].addAction({'verb': 'hit','object': 'Jerry'})
    actions['nop'] = world.agents['Tom'].addAction({'verb': 'doNothing'})
    actions['run'] = world.agents['Jerry'].addAction({'verb': 'run away'})
    actions['trick'] = world.agents['Jerry'].addAction({'verb': 'trick','object': 'Tom'})
    if order is None:
        order = ['Tom','Jerry']
    world.setOrder(order)
    return actions

def add_dynamics(world,actions):
    tree = makeTree({'distribution': [(approachMatrix(stateKey('Jerry','health'),0,.1),0.5),
                                    (noChangeMatrix(stateKey('Jerry','health')),0.5)]})
    world.setDynamics(stateKey('Jerry','health'),actions['hit'],tree)
    tree = makeTree({'distribution': [(approachMatrix(stateKey('Tom','health'),0,.1),0.5),
                                    (noChangeMatrix(stateKey('Tom','health')),0.5)]})
    world.setDynamics(stateKey('Tom','health'),actions['hit'],tree)


def add_reward(world):
    world.agents['Tom'].setReward(minimizeFeature(stateKey('Jerry','health'),'Tom'),1)
    world.agents['Jerry'].setReward(maximizeFeature(stateKey('Jerry','health'),'Jerry'),1)

def add_models(world,rationality=1.):
    model = next(iter(world.agents['Tom'].models.keys()))
    world.agents['Tom'].addModel('friend',rationality=rationality,parent=model)
    world.agents['Tom'].setReward(maximizeFeature(stateKey('Jerry','health'),'Jerry'),1,'friend')
    world.agents['Tom'].addModel('foe',rationality=rationality,parent=model)
    world.agents['Tom'].setReward(minimizeFeature(stateKey('Jerry','health'),'Jerry'),1,'foe')

def add_beliefs(world):
    for agent in world.agents.values():
        agent.resetBelief()
        agent.omega = [key for key in world.state.keys() if not isModelKey(key) and not isRewardKey(key)]

def test_conjunction():
    world = setup_world()
    add_state(world)
    actions = add_actions(world,['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom','health'),50) & thresholdRow(stateKey('Jerry','health'),50),
        True: incrementMatrix(stateKey('Jerry','health'),-5),
        False: noChangeMatrix(stateKey('Jerry','health'))})
    assert tree.branch.isConjunction
    world.setDynamics(stateKey('Jerry','health'),actions['hit'],tree)
    health = [world.getState('Jerry','health',unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == health[-2]
    world.setState('Tom','health',51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == health[-2]
    world.setState('Jerry','health',51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] < health[-2]

def test_disjunction():
    delta = 5
    threshold = 50
    world = setup_world()
    add_state(world)
    actions = add_actions(world,['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom','health'), threshold) | thresholdRow(stateKey('Jerry','health'), threshold),
        True: incrementMatrix(stateKey('Jerry','health'),-delta),
        False: noChangeMatrix(stateKey('Jerry','health'))})
    assert not tree.branch.isConjunction
    world.setDynamics(stateKey('Jerry','health'),actions['hit'],tree)
    health = [world.getState('Jerry','health',unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == health[-2]
    world.setState('Jerry','health', threshold+1)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == threshold-delta+1
    world.setState('Tom','health', threshold+1)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == threshold-2*delta+1

def test_greater_than():
    world = setup_world()
    add_state(world)
    actions = add_actions(world,['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom','health'),50),
        True: incrementMatrix(stateKey('Jerry','health'),-5),
        False: noChangeMatrix(stateKey('Jerry','health'))})
    world.setDynamics(stateKey('Jerry','health'),actions['hit'],tree)
    health = [world.getState('Jerry','health',unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] == health[-2]
    world.setState('Tom','health',51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry','health',unique=True))
    assert health[-1] < health[-2]

def dont_test_belief_update():
    world = setup_world()
    add_state(world)
    actions = add_actions(world)
    add_dynamics(world,actions)
    add_reward(world)
    add_beliefs(world)
    add_models(world)
    world.setMentalModel('Jerry','Tom',Distribution({'friend': 0.5,'foe': 0.5}))

    world.step()