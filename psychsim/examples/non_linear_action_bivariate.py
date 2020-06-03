import numpy as np
from psychsim.agent import Agent
from psychsim.helper_functions import get_bivariate_samples, tree_from_bivariate_samples
from psychsim.pwl import makeTree
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'An example of how to use the helper functions to create non-linear action dynamics in PsychSim.' \
                  'The input is a non-linear bivariate function, and the helper functions automatically generate' \
                  'two intertwined binary search trees that approximate the function.' \
                  'Test samples are then taken to estimate the approximation error.'

# parameters
NUM_SAMPLES = 101
MIN_X = -20
MAX_X = 20
MIN_Y = -10
MAX_Y = 10
NUM_TEST_SAMPLES = 1000
SEED = 0


def run_univariate_function(name, symbol_fmt, func):
    print('\n*************************************')
    print('Testing {} function'.format(name))

    # PsychSim elements
    world = World()
    agent = Agent('The Agent')
    world.addAgent(agent)

    # gets samples from real non-linear function
    x_params, y_params, sample_values = \
        get_bivariate_samples(func, MIN_X, MAX_X, MIN_Y, MAX_Y, NUM_SAMPLES, NUM_SAMPLES)
    sample_mean = np.nanmean(sample_values)

    # create two features: one holding the variable, the other the result (dependent)
    var_x = world.defineState(agent.name, 'var_x', float, lo=MIN_X, hi=MAX_X)
    var_y = world.defineState(agent.name, 'var_y', float, lo=MIN_Y, hi=MAX_Y)
    result = world.defineState(agent.name, 'result', float, lo=np.min(sample_values), hi=np.max(sample_values))
    world.setFeature(result, 0)

    # create action that is approximates the function, storing the result in the result feature
    action = agent.addAction({'verb': 'operation', 'action': name})
    tree = makeTree(tree_from_bivariate_samples(result, var_x, var_y, x_params, y_params, sample_values))
    world.setDynamics(result, action, tree)

    world.setOrder([agent.name])

    np.random.seed(SEED)
    values_original = []
    values_approx = []
    for i in range(NUM_TEST_SAMPLES):
        # gets random sample parameters
        x = MIN_X + np.random.rand() * (MAX_X - MIN_X)
        y = MIN_Y + np.random.rand() * (MAX_Y - MIN_Y)

        # sets variable and updates result
        world.setFeature(var_x, x)
        world.setFeature(var_y, y)
        world.step()

        real = func(x, y)
        psych = world.getValue(result)

        print('{:3}: {:30} | Expected: {:10.2f} | PsychSim: {:10.2f}'.format(i, symbol_fmt.format(x, y), real, psych))
        values_original.append(real)
        values_approx.append(psych)

    # gets error stats
    rmse = np.sqrt(np.mean((np.array(values_approx) - values_original) ** 2))
    print('=====================================')
    print('RMSE      = {:.3f}'.format(rmse))
    print('\nPress \'Enter\' to continue...')
    input()


if __name__ == '__main__':
    # some bivariate non-linear functions
    run_univariate_function('multiply', '{:.2f}*{:.2f}', np.multiply)
    run_univariate_function('division', '{:.2f}/{:.2f}', lambda x, y: x / y if y != 0 else 0.)
    run_univariate_function('logistic', 'logistic({:.2f}, {:.2f})',
                            lambda x, y: 1. / (1. + np.exp(-y * x)))
