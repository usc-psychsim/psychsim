import random
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import rewardKey, setToConstantMatrix, makeTree, actionKey, equalRow, WORLD, setTrueMatrix, \
    setFalseMatrix, trueRow
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of modeling partial observability in PsychSim. It models the classic POMDP "Tiger ' \
                  'problem", where a tiger is hidden behind two doors (left and right) which the agent cannot ' \
                  'observe. The agent can however perform a "listen" action that gives clues of where the tiger ' \
                  'might be located. The agent has an initial belief about the tiger\'s location which gets updated ' \
                  'after each "listen" action, where the agent also incurs in a small penalty. The agent can also ' \
                  'open the left or right door to "see" if the tiger is behind that door, receiving a penalty for ' \
                  'seeing the tiger, and a reward otherwise. After opening the door, the tiger is located behind ' \
                  'one of the doors at random.'

# feature values
HEAR_RIGHT_OBS = 'hear_right'
HEAR_LEFT_OBS = 'hear_left'
TIGER_RIGHT = 'tiger_right'
TIGER_LEFT = 'tiger_left'

# agent parameters
HORIZON = 1
DISCOUNT = 1
ACTION_SELECTION = 'random'  # untie best actions at random
MAX_STEPS = 20

SEED = 17


def _get_belief(feature: str, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(agent.name, unique=True)
    return world.getFeature(feature, state=agent.getBelief(model=model))


if __name__ == '__main__':
    random.seed(SEED)

    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # set agent parameters
    agent.setAttribute('discount', DISCOUNT)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', ACTION_SELECTION)

    # add hidden feature (tiger location)
    tiger_loc = world.defineState(WORLD, 'tiger_location', list, [TIGER_LEFT, TIGER_RIGHT])
    world.setFeature(tiger_loc, TIGER_LEFT)

    # add observations (hear and see tiger)
    hear_tiger_obs = world.defineState(agent.name, 'hear_tiger', list, [HEAR_LEFT_OBS, HEAR_RIGHT_OBS])
    world.setFeature(hear_tiger_obs, HEAR_LEFT_OBS)
    see_tiger_obs = world.defineState(agent.name, 'see_tiger', bool)
    world.setFeature(see_tiger_obs, False)

    # add actions

    # listen: 15% chance the agent hears the tiger behind the left door while it is actually behind
    # the right door and vice versa
    listen_act = agent.addAction({'verb': 'listen'})
    tree = {'if': equalRow(tiger_loc, TIGER_LEFT),
            True: Distribution({setToConstantMatrix(hear_tiger_obs, HEAR_LEFT_OBS): 0.85,
                                setToConstantMatrix(hear_tiger_obs, HEAR_RIGHT_OBS): 0.15}),
            False: Distribution({setToConstantMatrix(hear_tiger_obs, HEAR_LEFT_OBS): 0.15,
                                 setToConstantMatrix(hear_tiger_obs, HEAR_RIGHT_OBS): 0.85})}
    world.setDynamics(hear_tiger_obs, listen_act, makeTree(tree))

    # after a door is opened, the problem is reset (the tiger is randomly assigned to a door with 50/50 chance)
    tree = makeTree(Distribution({setToConstantMatrix(tiger_loc, TIGER_LEFT): 0.5,
                                  setToConstantMatrix(tiger_loc, TIGER_RIGHT): 0.5}))
    open_left_act = agent.addAction({'verb': 'open_left_door'})
    world.setDynamics(tiger_loc, open_left_act, makeTree(tree))
    open_right_act = agent.addAction({'verb': 'open_right_door'})
    world.setDynamics(tiger_loc, open_right_act, makeTree(tree))

    # agent sees tiger if opens the door where the agent is
    tree = {'if': equalRow(tiger_loc, TIGER_LEFT),
            True: setTrueMatrix(see_tiger_obs),
            False: setFalseMatrix(see_tiger_obs)}
    world.setDynamics(see_tiger_obs, open_left_act, makeTree(tree))

    tree = {'if': equalRow(tiger_loc, TIGER_RIGHT),
            True: setTrueMatrix(see_tiger_obs),
            False: setFalseMatrix(see_tiger_obs)}
    world.setDynamics(see_tiger_obs, open_right_act, makeTree(tree))

    # set reward function: -1 for listening, -100 if agent sees tiger, 10 if agent does not see tiger
    rwd_key = rewardKey(agent.name)
    tree = {'if': equalRow(actionKey(agent.name), listen_act),
            True: setToConstantMatrix(rwd_key, -1.),
            False: {'if': trueRow(see_tiger_obs),
                    True: setToConstantMatrix(rwd_key, -100.),
                    False: setToConstantMatrix(rwd_key, 10.)}}
    agent.setReward(makeTree(tree), 1.)

    # set order
    world.setOrder([agent.name])

    agent.set_observations(unobservable=[tiger_loc])  # agent does not directly observe tiger's position
    agent.setBelief(tiger_loc, Distribution({TIGER_LEFT: 0.5, TIGER_RIGHT: 0.5}))  # initial belief

    print('====================================')
    print(f'Initial beliefs:\n{_get_belief(tiger_loc)}')
    print(f'Initial loc: {world.getFeature(tiger_loc, unique=True)}')
    print(f'Reward function:\n{agent.getReward(agent.get_true_model())}')

    for i in range(MAX_STEPS):
        step = world.step(select=True)

        print('====================================')
        print(f'Action: {world.getFeature(actionKey(agent.name), unique=True)}')
        print('Observations:')
        print(f'\tHear tiger: {world.getFeature(hear_tiger_obs, unique=True)}')
        print(f'\tSee tiger: {world.getFeature(see_tiger_obs, unique=True)}')
        print(f'Updated belief:\n{_get_belief(tiger_loc)}')
        print(f'Current loc: {world.getFeature(tiger_loc, unique=True)}')
        print(f'Reward: {agent.reward(model=agent.get_true_model())}')
