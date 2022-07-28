import copy
import math

from psychsim.probability import Distribution
from psychsim.action import ActionSet
from psychsim.world import World
from psychsim.pwl.keys import makeFuture, stateKey
from psychsim.pwl.matrix import approachMatrix, incrementMatrix, noChangeMatrix, setTrueMatrix
from psychsim.pwl.plane import equalRow, thresholdRow, trueRow
from psychsim.pwl.tree import makeTree
from psychsim.reward import minimizeFeature, maximizeFeature


def setup_world():
    # Create world
    world = World()
    # Create agents
    world.addAgent('Tom')
    world.addAgent('Jerry')
    return world


def add_state(world):
    """Create state features"""
    world.defineState('Tom', 'health', int, lo=0, hi=100,
                      description='Tom\'s wellbeing')
    world.setState('Tom', 'health', 50)
    world.defineState('Jerry', 'health', int, lo=0, hi=100,
                      description='Jerry\'s wellbeing')
    world.setState('Jerry', 'health', 50)


def add_actions(world, order=None):
    """Create actions"""
    actions = {}
    actions['chase'] = world.agents['Tom'].addAction({'verb': 'chase', 'object': 'Jerry'})
    actions['hit'] = world.agents['Tom'].addAction({'verb': 'hit', 'object': 'Jerry'})
    actions['nop'] = world.agents['Tom'].addAction({'verb': 'doNothing'})
    actions['run'] = world.agents['Jerry'].addAction({'verb': 'run away'})
    actions['trick'] = world.agents['Jerry'].addAction({'verb': 'trick', 'object': 'Tom'})
    if order is None:
        order = ['Tom', 'Jerry']
    world.setOrder(order)
    return actions


def add_dynamics(world, actions):
    tree = makeTree({'distribution': [(approachMatrix(stateKey('Jerry', 'health'), .1, 0), 0.5),
                                      (noChangeMatrix(stateKey('Jerry', 'health')), 0.5)]})
    world.setDynamics(stateKey('Jerry', 'health'), actions['hit'], tree)
    tree = makeTree({'distribution': [(approachMatrix(stateKey('Tom', 'health'), .1, 1), 0.5),
                                      (noChangeMatrix(stateKey('Tom', 'health')),  0.5)]})
    world.setDynamics(stateKey('Tom', 'health'), actions['hit'], tree)


def add_trick_dynamics(world, actions):
    trick = world.define_state('Tom', 'tricked', bool, default=False)
    world.setDynamics(trick, actions['trick'], makeTree(setTrueMatrix(trick)))
    tree = makeTree({'if': trueRow(makeFuture(trick)),
                     # If tricked, bad
                     True: Distribution({approachMatrix(stateKey('Tom', 'health'), .1, 0): 0.5,
                                         noChangeMatrix(stateKey('Tom', 'health')): 0.5}),
                     # If not tricked, good
                     False: Distribution({approachMatrix(stateKey('Tom', 'health'), .1, 1): 0.5,
                                          noChangeMatrix(stateKey('Tom', 'health')): 0.5})})
    world.setDynamics(stateKey('Tom', 'health'), actions['hit'], tree)
    world.agents['Tom'].setReward(maximizeFeature(stateKey('Tom', 'health'), 'Tom'), 1)
    tree = makeTree({'if': trueRow(makeFuture(trick)),
                     # If tricked, good for Jerry
                     True: Distribution({approachMatrix(stateKey('Jerry', 'health'), .1, 1): 0.5,
                                         noChangeMatrix(stateKey('Jerry', 'health')): 0.5}),
                     # If not tricked, bad for Jerry
                     False: Distribution({approachMatrix(stateKey('Jerry', 'health'), .1, 0): 0.5,
                                          noChangeMatrix(stateKey('Jerry', 'health')): 0.5})})
    world.setDynamics(stateKey('Jerry', 'health'), actions['hit'], tree)


def add_reward(world):
    world.agents['Tom'].setReward(minimizeFeature(stateKey('Jerry', 'health'), 'Tom'), 1)
    world.agents['Jerry'].setReward(maximizeFeature(stateKey('Jerry', 'health'), 'Jerry'), 1)


def add_models(world, rationality=1):
    model = world.agents['Tom'].get_true_model()
    world.agents['Tom'].addModel('friend', rationality=rationality, selection='distribution', horizon=1, parent=model)
    world.agents['Tom'].setReward(maximizeFeature(stateKey('Jerry', 'health'), 'Tom'), 1, 'friend')
    world.agents['Tom'].create_belief_state(model='friend')
    world.agents['Tom'].addModel('foe', rationality=rationality, selection='distribution', horizon=1, parent=model)
    world.agents['Tom'].setReward(minimizeFeature(stateKey('Jerry', 'health'), 'Tom'), 1, 'foe')
    world.agents['Tom'].create_belief_state(model='foe')
    zero = world.agents['Jerry'].zero_level()
    world.setModel('Jerry', zero, 'Tom', 'friend')
    world.setModel('Jerry', zero, 'Tom', 'foe')


def add_beliefs(world):
    agents = list(world.agents.values())
    for index, agent in enumerate(agents):
        agent.create_belief_state()
        agent.set_observations()


def test_dynamics():
    world = setup_world()
    add_state(world)
    actions = add_actions(world, ['Tom'])
    add_dynamics(world, actions)
    health = [world.getState('Jerry', 'health', unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health'))
    assert len(health[1]) == 2
    assert health[1][health[0]] == 0.5
    assert health[1][.9*health[0]] == 0.5
    # No null subdistributions in state
    for dist in world.state.distributions.values():
        assert len(dist) > 0
    

def test_legality():
    world = setup_world()
    add_state(world)
    actions = add_actions(world, ['Tom'])
    action = actions['hit']
    value = world.getState('Tom', 'health', unique=True)
    tree = makeTree({'if': thresholdRow(stateKey('Tom', 'health'), value-5), True: True, False: False})
    world.agents['Tom'].setLegal(action, tree)
    assert action in world.agents['Tom'].getLegalActions()
    tree = makeTree({'if': thresholdRow(stateKey('Tom', 'health'), value+5), True: True, False: False})
    world.agents['Tom'].setLegal(action, tree)
    assert action not in world.agents['Tom'].getLegalActions()


def test_default_branch():
    world = setup_world()
    add_state(world)
    mood = world.defineState('Tom', 'mood', list, ['happy', 'neutral', 'sad', 'angry'])
    world.setFeature(mood, 'angry')
    actions = add_actions(world, ['Tom'])
    tree = makeTree({'if': equalRow(mood, ['happy', 'neutral']),
                     0: incrementMatrix(stateKey('Tom', 'health'), 1),
                     1: noChangeMatrix(stateKey('Tom', 'health')),
                     None: incrementMatrix(stateKey('Tom', 'health'), -1)})
    world.setDynamics(stateKey('Tom', 'health'), actions['hit'], tree)
    health = [world.getState('Tom', 'health', unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Tom', 'health', unique=True))
    assert health[-1] == health[-2] - 1


def test_conjunction():
    world = setup_world()
    add_state(world)
    actions = add_actions(world,['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom', 'health'), 50) & thresholdRow(stateKey('Jerry', 'health'), 50),
                     True: incrementMatrix(stateKey('Jerry', 'health'), -5),
                     False: noChangeMatrix(stateKey('Jerry', 'health'))})
    assert tree.branch.isConjunction
    world.setDynamics(stateKey('Jerry', 'health'), actions['hit'], tree)
    health = [world.getState('Jerry', 'health', unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == health[-2]
    for dist in world.state.distributions.values():
        assert len(dist) > 0
    world.setState('Tom', 'health', 51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == health[-2]
    for dist in world.state.distributions.values():
        assert len(dist) > 0
    world.setState('Jerry', 'health', 51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] < health[-2]
    for dist in world.state.distributions.values():
        assert len(dist) > 0

    hi = 75
    lo = 25
    hi_prob = 0.25
    world.setState('Tom', 'health', Distribution({hi: hi_prob, lo: 1-hi_prob}))
    world.setState('Jerry', 'health', Distribution({hi: hi_prob, lo: 1-hi_prob}))
    world.step({'Tom': actions['hit']})
    dist = world.getState('Jerry', 'health')
    assert dist[hi-5] == hi_prob**2
    assert dist[hi] == hi_prob*(1-hi_prob)
    assert dist[lo] == 1-hi_prob
    for dist in world.state.distributions.values():
        assert len(dist) > 0


def test_disjunction():
    delta = 5
    threshold = 50
    world = setup_world()
    add_state(world)
    actions = add_actions(world, ['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom', 'health'), threshold) | thresholdRow(stateKey('Jerry', 'health'), threshold),
                     True: incrementMatrix(stateKey('Jerry', 'health'),-delta),
                     False: noChangeMatrix(stateKey('Jerry', 'health'))})
    assert not tree.branch.isConjunction
    world.setDynamics(stateKey('Jerry', 'health'), actions['hit'],tree)
    health = [world.getState('Jerry', 'health', unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == health[-2]
    world.setState('Jerry', 'health', threshold+1)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == threshold-delta+1
    world.setState('Tom', 'health', threshold+1)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == threshold-2*delta+1


def test_greater_than():
    world = setup_world()
    add_state(world)
    actions = add_actions(world, ['Tom'])
    tree = makeTree({'if': thresholdRow(stateKey('Tom', 'health'), 50),
                     True: incrementMatrix(stateKey('Jerry', 'health'), -5),
                     False: noChangeMatrix(stateKey('Jerry', 'health'))})
    world.setDynamics(stateKey('Jerry', 'health'), actions['hit'], tree)
    health = [world.getState('Jerry', 'health', unique=True)]
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] == health[-2]
    world.setState('Tom', 'health',51)
    world.step({'Tom': actions['hit']})
    health.append(world.getState('Jerry', 'health', unique=True))
    assert health[-1] < health[-2]


def test_reward():
    world = setup_world()
    add_state(world)
    add_reward(world)
    add_models(world)
    assert world.agents['Tom'].reward() == -world.getState('Jerry', 'health', unique=True)
    assert world.agents['Tom'].reward(model='foe') == -world.getState('Jerry', 'health', unique=True)
    assert world.agents['Tom'].reward(model='friend') == world.getState('Jerry', 'health', unique=True)


def test_belief_update():
    world = setup_world()
    add_state(world)
    actions = add_actions(world)
    add_dynamics(world, actions)
    add_reward(world)
    add_beliefs(world)
    add_models(world)
    jerry = world.agents['Jerry']
    world.setModel('Tom', Distribution({'friend': 0.5, 'foe': 0.5}), 'Jerry', jerry.get_true_model())
    world.step({'Tom': actions['hit']})
#    for model, belief in jerry.getBelief().items():
#        print(model)
#        print(world.getModel('Tom', belief))


def test_trick():
    world = setup_world()
    add_state(world)
    actions = add_actions(world, order=['Jerry', 'Tom'])
    add_dynamics(world, actions)
    add_trick_dynamics(world, actions)
    add_reward(world)
    world.step({'Jerry': actions['trick']})
    tom = world.agents['Tom']
    assert tom.decide(model=tom.get_true_model())['action'] == actions['chase']
    world.setState('Tom', 'tricked', False)
    assert tom.decide(model=tom.get_true_model())['action'] == actions['hit']


def test_zero_level():
    world = setup_world()
    add_state(world)
    actions = add_actions(world, [{'Tom', 'Jerry'}])
    add_trick_dynamics(world, actions)
    add_reward(world)
    jerry = world.agents['Jerry']
    tom = world.agents['Tom']
    jerry0 = jerry.zero_level(horizon=1)
    R = jerry.getReward(jerry.get_true_model())   
    R0 = jerry.getReward(jerry0)
    assert R == R0 
    decision = jerry.decide(model=jerry0, others={tom.name: actions['hit']}, debug={'preserve_states': True})
    assert decision['action'] == actions['trick']
    tom1 = tom.n_level(n=1, models={jerry.name: jerry0})
    decision = tom.decide(model=tom1, others={jerry.name: actions['trick']})
    assert decision['action'] == actions['chase']
    decision = tom.decide(model=tom1, others={jerry.name: actions['run']})
    assert decision['action'] == actions['hit']


def test_selection():
    world = setup_world()
    add_state(world)
    actions = add_actions(world)
    add_dynamics(world, actions)
    add_reward(world)
    add_models(world)
    tom = world.agents['Tom']
    model = tom.get_true_model()
    result = tom.decide()
    verify_decision(result, model, ActionSet)
    result = tom.decide(selection='softmax')
    verify_decision(result, model, ActionSet)
    result = tom.decide(selection='distribution')
    verify_decision(result, model, Distribution, len(tom.actions))


def verify_decision(result, model: str, cls, length=None):
    assert len(result) == 3
    assert model in result
    assert 'policy' in result
    assert 'probability' in result
    decision = result[model]['action']
    assert isinstance(decision, cls)
    if length is not None:
        assert len(decision) == length


def test_step_select():
    world = setup_world()
    add_state(world)
    actions = add_actions(world)
    add_dynamics(world, actions)
    add_reward(world)
    add_models(world)
    tom = world.agents['Tom']
    tom.setAttribute('strict_max', False, tom.get_true_model())
    tom.setAttribute('sample', False, tom.get_true_model())
    tom.setAttribute('tiebreak', False, tom.get_true_model())
    # Test full simulation step
    state = copy.deepcopy(world.state)
    prob = world.step(state=state)
    assert math.isclose(prob, 1)
    # Test sampling an outcome
    state = copy.deepcopy(world.state)
    prob = world.step(state=state, select=True)
    assert len(state) == 1
    action = world.getAction(tom.name, state, True)
    if action == actions['hit']:
        assert math.isclose(prob, 0.246675822760567)
    else:
        raise NotImplementedError(f'Have not recorded whether unlikely probability is {prob}')
    # Test selection of most likely outcome
    state = copy.deepcopy(world.state)
    prob = world.step(state=state, select='max')
    assert math.isclose(prob, 0.246675822760567)
    # Test sampling in decision-making
    tom.setAttribute('sample', True, tom.get_true_model())
    state = copy.deepcopy(world.state)
    prob = world.step(state=state)
    action = world.getAction(tom.name, state, True)
    table = {'chase': 0.006648354478866005,
             'doNothing': 0.006648354478866005,
             'hit': 0.9867032910422682}
    assert math.isclose(prob, table[action['verb']])
