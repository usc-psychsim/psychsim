from psychsim.action import ActionSet
from psychsim.agent import Agent
from psychsim.helper_functions import multi_set_matrix, get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, CONSTANT, isStateKey, setToFeatureMatrix, actionKey
from psychsim.reward import maximizeFeature
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example that models one agent navigating left or right along one dimension (one position feature). ' \
                  'We set a belief to the agent for an incorrect position, and see the agent act based on that belief, ' \
                  'i.e., in its mind the position is changing, but is not aligned with the real/true position.'

# parameters
HORIZON = 3
DISCOUNT = 1
MAX_STEPS = 3

if __name__ == '__main__':

    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # set parameters
    agent.setAttribute('discount', DISCOUNT)
    agent.setHorizon(HORIZON)

    # add position variable
    pos = world.defineState(agent.name, 'position', int, lo=-100, hi=100)
    world.setFeature(pos, 0)

    # define agents' actions (stay 0, left -1 and right +1)
    action = agent.addAction({'verb': 'move', 'action': 'nowhere'})
    tree = makeTree(setToFeatureMatrix(pos, pos))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'left'})
    tree = makeTree(multi_set_matrix(pos, {pos: 1, CONSTANT: -1}))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(multi_set_matrix(pos, {pos: 1, CONSTANT: 1}))
    world.setDynamics(pos, action, tree)

    # define rewards (maximize position, i.e., always go right)
    agent.setReward(maximizeFeature(pos, agent.name), 1)

    # set order
    world.setOrder([agent.name])

    # agent has initial beliefs about its position, which will be updated after executing actions
    agent.omega = {actionKey(agent.name)}  # todo should not need this
    agent.setBelief(pos, Distribution({10: 0.5, 12: 0.5}))
    # agent.setBelief(pos, 10, get_true_model_name(agent))

    print('====================================')
    print('Initial beliefs:')
    world.printBeliefs(agent.name)

    for i in range(MAX_STEPS):
        print('====================================')
        print('Current pos: {0}'.format(world.getValue(pos)))

        # decision: left, right or no-move?
        step = world.step()

        # prints all models and beliefs
        print('____________________________________')
        print("Updated beliefs:")
        world.printBeliefs(agent.name)
        print('____________________________________')

        # print step
        # world.explain(step, level=2) # todo cannot retrieve old 'outcomes' from step
