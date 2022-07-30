import random
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, modelKey, incrementMatrix, setToConstantMatrix, rewardKey, actionKey
from psychsim.reward import maximizeFeature, minimizeFeature

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of an agent (the actor) moves in the environment (with stochastic actions).'

# parameters
MAX_STEPS = 10
AGENT_NAME = 'actor'
HORIZON = 2
AGENT_SELECTION = 'distribution'
SEED = 17

if __name__ == '__main__':
    random.seed(SEED)

    # create world and add actor agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', AGENT_SELECTION)

    # add location variable
    loc = world.defineState(agent.name, 'location', int, -100, 100)
    world.setFeature(loc, 0)

    # define agents' actions (left and right)
    left_action = agent.addAction({'verb': 'move', 'action': 'left'})
    tree = makeTree(incrementMatrix(loc, -1))
    world.setDynamics(loc, left_action, tree)
    right_action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(incrementMatrix(loc, 1))
    world.setDynamics(loc, right_action, tree)

    # define true reward (maximize loc)
    agent.setReward(maximizeFeature(loc, agent.name), 1)

    world.setOrder([{agent.name}])

    print('====================================')
    print(f'Initial loc: {world.getFeature(loc)}')

    left_action_value = world.value2float(actionKey(agent.name), left_action)
    right_action_value = world.value2float(actionKey(agent.name), right_action)

    action = [(setToConstantMatrix(actionKey(agent.name), left_action_value), 1),
              (setToConstantMatrix(actionKey(agent.name), right_action_value), 0)]
    stochastic_action = {agent.name: makeTree({'distribution': action})}

    deterministic_action = {agent.name: left_action}

    print(action)
    for i in range(MAX_STEPS):
        print('====================================')
        step = world.step(actions=stochastic_action, select=False, horizon=HORIZON, tiebreak='distribution')
        print(f'Current loc: {world.getFeature(loc)}')
