import argparse
import time
import numpy as np
from vprof import runner
from psychsim.agent import Agent
from psychsim.helper_functions import multi_set_matrix, get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import makeTree
from psychsim.reward import maximizeFeature
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'This script allows test profiling PsychSim. We can define the number of agents simulated, the number' \
                  'of actions available, planning horizon, etc. If we plug in the "vprof" profiler then we can check ' \
                  'the portions of the code contributing the most to CPU time and memory.'

# parameters
SEED = 0  # to get consistent results
NUM_TIME_STEPS = 4
NUM_AGENTS = 2  # seems to have an exponential effect
HORIZON = 3  # seems to have a polynomial effect
NUM_ACTIONS_PER_AGENT = 3  # seems to have a polynomial effect
NUM_FEATURES_PER_AGENT = 30  # seems to have a linear effect
NUM_FEATURES_ACTION = 10  # the lower the higher the number of trees per action and dependencies (does not have much effect)
MAX_FEATURE_VALUE = 10

PARALLEL = False
RUN_PROFILER = False


def get_fake_model_name(agent):
    return 'fake {} model'.format(agent.name)


def run():
    global args
    world = setup()

    print('====================================')
    step = None
    start_time = time.time()
    total_time = 0
    for i in range(0, args.steps):
        print('Step {}...'.format(i))
        start_clock = time.perf_counter()
        step = world.step()
        step_time = time.perf_counter() - start_clock
        total_time += step_time
        print('Clock time: {:.2f}s'.format(step_time))
        world.printState()
        print('____________________________________')

    print('Total time: {:.2f}s'.format(time.time() - start_time))
    print('Avg. time: {:.2f}s'.format(total_time / args.steps))
    print('____________________________________')
    # world.explain(step, level=2)  # todo step does not provide outcomes anymore


def setup():
    global args

    np.random.seed(args.seed)
    # create world and add agents
    world = World()
    world.memory = False
    world.parallel = args.parallel
    agents = []
    agent_features = {}
    for ag in range(args.agents):
        agent = Agent('Agent' + str(ag))
        world.addAgent(agent)
        agents.append(agent)

        # set agent's params
        agent.setAttribute('discount', 1)
        agent.setHorizon(args.horizon)

        # add features, initialize at random
        features = []
        agent_features[agent] = features
        for f in range(args.features_agent):
            feat = world.defineState(agent.name, 'Feature{}'.format(f), int, lo=0, hi=1000)
            world.setFeature(feat, np.random.randint(0, MAX_FEATURE_VALUE))
            features.append(feat)

        # set random reward function
        agent.setReward(maximizeFeature(np.random.choice(features), agent.name), 1)

        # add mental copy of true model and make it static (we do not have beliefs in the models)
        agent.addModel(get_fake_model_name(agent), parent=get_true_model_name(agent))
        agent.setAttribute('static', True, get_fake_model_name(agent))

        # add actions
        for ac in range(args.actions):
            action = agent.addAction({'verb': '', 'action': 'Action{}'.format(ac)})
            i = ac
            while i + args.features_action < args.features_agent:

                weights = {}
                for j in range(args.features_action):
                    weights[features[i + j + 1]] = 1
                tree = makeTree(multi_set_matrix(features[i], weights))
                world.setDynamics(features[i], action, tree)

                i += args.features_action

    # define order
    world.setOrder([set(ag.name for ag in agents)])

    for agent in agents:
        # test belief update:
        # - set a belief in one feature to the actual initial value (should not change outcomes)
        # world.setModel(agent.name, Distribution({True: 1.0}))
        rand_feat = np.random.choice(agent_features[agent])
        agent.setBelief(rand_feat, world.getValue(rand_feat))
        print('{} will always observe {}={}'.format(agent.name, rand_feat, world.getValue(rand_feat)))

    # set mental model of each agent in all other agents
    for i in range(args.agents):
        for j in range(i + 1, args.agents):
            world.setMentalModel(agents[i].name, agents[j].name, Distribution({get_fake_model_name(agents[j]): 1}))
            world.setMentalModel(agents[j].name, agents[i].name, Distribution({get_fake_model_name(agents[i]): 1}))

    return world


def main():
    global args
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('-t', '--steps', type=int, help='Number of timesteps to simulate.', default=NUM_TIME_STEPS)
    parser.add_argument('-a', '--agents', type=int, help='Number of agents to create.', default=NUM_AGENTS)
    parser.add_argument('-b', '--actions', type=int, help='Number of actions per agent.', default=NUM_ACTIONS_PER_AGENT)
    parser.add_argument('-fa', '--features-agent', type=int,
                        help='Number of features associated with each agent.', default=NUM_FEATURES_PER_AGENT)
    parser.add_argument('-fb', '--features-action', type=int,
                        help='Number of features used by each action.', default=NUM_FEATURES_ACTION)
    parser.add_argument('-ph', '--horizon', type=int, help='Planning horizon.', default=HORIZON)
    parser.add_argument('-s', '--seed', type=int,
                        help='The seed used to initialized the random number generator.', default=SEED)
    parser.add_argument('-p', '--parallel', help='Whether to use multiple processes to run the simulation.',
                        action='store_true')
    parser.add_argument('--profiler', help='Whether to run the performance profiler vprof. Note: '
                                           'run run \'vprof -r\' in command line first to create a listener.',
                        action='store_true')
    args = parser.parse_args()
    if args.profiler:
        # NOTE: run 'vprof -r' in command line first to create a listener
        # runner.run(run, 'cmhp', host='localhost', port=8000)
        runner.run(run, 'c', host='localhost', port=8000)
    else:
        run()


if __name__ == '__main__':
    global args
    main()
