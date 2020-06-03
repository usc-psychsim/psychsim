import numpy as np
from psychsim.agent import Agent
from psychsim.helper_functions import get_univariate_samples, tree_from_univariate_samples
from psychsim.pwl import makeTree
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'An example of how to use the helper functions to create non-linear action dynamics in PsychSim.' \
                  'The input is a non-linear univariate function, and the helper functions automatically generate' \
                  'a binary search tree that approximates the function.' \
                  'Test samples are then taken to estimate the approximation error.'

# parameters
NUM_SAMPLES = 101
MIN_X = -20
MAX_X = 20
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
    x_params, sample_values = get_univariate_samples(func, MIN_X, MAX_X, NUM_SAMPLES)
    sample_mean = np.nanmean(sample_values)

    # create two features: one holding the variable, the other the result (dependent)
    var = world.defineState(agent.name, 'var', float, lo=MIN_X, hi=MAX_X)
    result = world.defineState(agent.name, 'result', float, lo=np.min(sample_values), hi=np.max(sample_values))
    world.setFeature(result, 0)

    # create action that is approximates the function, storing the result in the result feature
    action = agent.addAction({'verb': 'operation', 'action': name})
    tree = makeTree(tree_from_univariate_samples(result, var, x_params, sample_values))
    world.setDynamics(result, action, tree)

    world.setOrder([agent.name])

    np.random.seed(SEED)
    values_original = []
    values_approx = []
    for i in range(NUM_TEST_SAMPLES):
        # gets random sample parameter
        x = MIN_X + np.random.rand() * (MAX_X - MIN_X)

        # sets variable and updates result
        world.setFeature(var, x)
        world.step()

        real = func(x)
        psych = world.getValue(result)

        print('{:3}: {:15} | Expected: {:10.3f} | PsychSim: {:10.3f}'.format(i, symbol_fmt.format(x), real, psych))
        values_original.append(real)
        values_approx.append(psych)

    # gets error stats
    rmse = np.sqrt(np.nanmean((np.array(values_approx) - values_original) ** 2))
    print('=====================================')
    print('RMSE      = {:.3f}'.format(rmse))
    print('\nPress \'Enter\' to continue...')
    input()


if __name__ == '__main__':
    # some univariate non-linear functions
    run_univariate_function('root', '√({:.3f})', np.sqrt)
    run_univariate_function('log', 'log({:.3f})', np.log)
    run_univariate_function('cube', '{:.3f}^3', lambda x: x ** 3)
