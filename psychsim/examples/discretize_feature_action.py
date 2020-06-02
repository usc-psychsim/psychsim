import numpy as np
from psychsim.agent import Agent
from psychsim.helper_functions import discretization_tree
from psychsim.pwl import makeTree
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of how to discretize a feature\'s value with an action using a helper function. ' \
                  'Discretization occurs by having an action whose PWL dynamics approximate the value of the feature ' \
                  'to the closest bin from a discrete set of bins by performing a binary search.' \
                  'This method can be used to discretize features\' values at each timestep by having a "dummy" agent ' \
                  'that has a single action, i.e., the discretization action. The agent can be placed in the turn ' \
                  'order such that it acts *after* all other agents.' \
                  'A similar method can be used to have discretized rewards, i.e., to create a reward function whose ' \
                  'value is the discretized value of a feature.'

HIGH = 100
LOW = 50
NUM_BINS = 11
NUM_SAMPLES = 100

if __name__ == '__main__':

    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # add variable
    feat = world.defineState(agent.name, 'x', float, lo=LOW, hi=HIGH)

    # add single action that discretizes the feature
    action = agent.addAction({'verb': '', 'action': 'discretize'})
    tree = makeTree(discretization_tree(world, feat, NUM_BINS))
    world.setDynamics(feat, action, tree)

    world.setOrder([{agent.name}])

    print('====================================')
    print('High:\t{}'.format(HIGH))
    print('Low:\t{}'.format(LOW))
    print('Bins:\t{}'.format(NUM_BINS))

    print('\nSamples/steps:')
    values_original = []
    values_discrete = []
    for i in range(NUM_SAMPLES):
        num = np.random.uniform(LOW, HIGH)
        world.setFeature(feat, num)

        before = world.getValue(feat)
        world.step()
        after = world.getValue(feat)

        print('{:.3f}\t-> {}'.format(before, after))
        values_original.append(before)
        values_discrete.append(after)

    # calculates RMSE
    rmse = np.sqrt(np.mean((np.array(values_discrete) - values_original) ** 2))
    print('\nRMSE: {:.3f}'.format(rmse))
