import numpy as np
from psychsim.agent import Agent
from psychsim.helper_functions import discretize_feature_in_place
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of how to discretize a feature\'s value in-place using a helper function. ' \
                  'Discretization occurs by directly approximating the value of the feature to the closest bin from ' \
                  'a discrete set of bins.' \
                  'This method can be used to discretize features\' values at each timestep after world.update().'

HIGH = 100
LOW = 50
NUM_BINS = 11
NUM_SAMPLES = 100

if __name__ == '__main__':

    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # add feature to world
    feat = world.defineState(agent.name, 'x', float, lo=LOW, hi=HIGH)

    print('====================================')
    print('High:\t{}'.format(HIGH))
    print('Low:\t{}'.format(LOW))
    print('Bins:\t{}'.format(NUM_BINS))

    print('\nSamples:')
    values_original = []
    values_discrete = []
    for i in range(NUM_SAMPLES):
        num = np.random.uniform(LOW, HIGH)
        world.setFeature(feat, num)

        before = world.getValue(feat)
        discretize_feature_in_place(world, feat, NUM_BINS)
        after = world.getValue(feat)

        print('{:.3f}\t-> {}'.format(before, after))
        values_original.append(before)
        values_discrete.append(after)

        # calculates RMSE
    rmse = np.sqrt(np.mean((np.array(values_discrete) - values_original) ** 2))
    print('\nRMSE: {:.3f}'.format(rmse))
