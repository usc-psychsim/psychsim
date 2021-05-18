import logging
import random
from psychsim.agent import Agent
from psychsim.helper_functions import set_illegal_action, set_legal_action
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, setToConstantMatrix, rewardKey, equalRow, equalFeatureRow, modelKey

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of how to set legality for agent\'s models in PsychSim using the helper functions. ' \
                  'There are two agents how have two actions: go left or go right. If the agents choose the same ' \
                  'action, they both receive a penalty, if they go to different directions they both receive a ' \
                  'reward. We create mental models of each agent and then add them to the other. Then we "remove" the' \
                  'right action of the *models* be setting it always illegal using the helper functions. This results' \
                  'in the agents always choosing to go right, because during ToM reasoning they believe that going ' \
                  'left is the only option available to the other. Without this restriction, any combination is ' \
                  'possible between the agents\' decisions because their models don\'t have preferences.'

# parameters (positive reward if sides are different, otherwise punishment)
DIFF_SIDES_RWD = 1
SAME_SIDE_RWD = -1
INVALID = 0

NUM_STEPS = 10
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# decision labels
NOT_DECIDED = 'none'
WENT_LEFT = 'left'
WENT_RIGHT = 'right'

DEBUG = False


def get_fake_model_name(agent):
    return f'fake_{agent.name}_model'


# defines reward tree
def get_reward_tree(agent, my_side, other_side):
    reward_key = rewardKey(agent.name)
    return makeTree({'if': equalRow(my_side, NOT_DECIDED),  # if I have not decided
                     True: setToConstantMatrix(reward_key, INVALID),
                     False: {'if': equalRow(other_side, INVALID),  # if other has not decided
                             True: setToConstantMatrix(reward_key, INVALID),
                             False: {'if': equalFeatureRow(my_side, other_side),  # if my_side == other_side
                                     True: setToConstantMatrix(reward_key, SAME_SIDE_RWD),
                                     False: setToConstantMatrix(reward_key, DIFF_SIDES_RWD)}}})


if __name__ == '__main__':

    random.seed(0)

    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create world and add agents
    world = World()
    agent1 = Agent('Agent 1')
    world.addAgent(agent1)
    agent2 = Agent('Agent 2')
    world.addAgent(agent2)

    sides = []
    rights = []
    lefts = []

    agents = [agent1, agent2]
    for agent in agents:
        # set agent's params
        agent.setAttribute('discount', 1)
        agent.setHorizon(1)
        agent.setAttribute('selection', TIEBREAK)

        # add 'side chosen' variable (0 = didn't decide, 1 = went left, 2 = went right)
        side = world.defineState(agent.name, 'side', list, [NOT_DECIDED, WENT_LEFT, WENT_RIGHT])
        world.setFeature(side, NOT_DECIDED)
        sides.append(side)

        # define agents' actions (left and right)
        action = agent.addAction({'verb': '', 'action': 'go left'})
        tree = makeTree(setToConstantMatrix(side, WENT_LEFT))
        world.setDynamics(side, action, tree)
        lefts.append(action)

        action = agent.addAction({'verb': '', 'action': 'go right'})
        tree = makeTree(setToConstantMatrix(side, WENT_RIGHT))
        world.setDynamics(side, action, tree)
        rights.append(action)

        # create a new model for the agent
        agent.addModel(get_fake_model_name(agent), parent=agent.get_true_model())

    # defines payoff matrices
    agent1.setReward(get_reward_tree(agent1, sides[0], sides[1]), 1)
    agent2.setReward(get_reward_tree(agent2, sides[1], sides[0]), 1)

    # define order
    world.setOrder([{agent1.name, agent2.name}])

    # add mental model of the other for each agent
    world.setMentalModel(agent1.name, agent2.name, Distribution({get_fake_model_name(agent2): 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({get_fake_model_name(agent1): 1}))

    # 'hides' right actions from models by setting them illegal
    # (therefore agents should always choose right because they think the other will choose left)
    set_illegal_action(agent1, rights[0], [get_fake_model_name(agent1)])
    set_illegal_action(agent2, rights[1], [get_fake_model_name(agent2)])

    # # ** unnecessary / just for illustration **: set left actions legal for both the agents and their models
    # set_legal_action(agent1, lefts[0], [agent1.get_true_model(), get_fake_model_name(agent1)])
    # set_legal_action(agent2, lefts[1], [agent2.get_true_model(), get_fake_model_name(agent2)])

    agent1.resetBelief(model=agent1.get_true_model())
    agent1.resetBelief(model=get_fake_model_name(agent1))
    agent2.resetBelief(model=agent2.get_true_model())
    agent2.resetBelief(model=get_fake_model_name(agent2))

    for t in range(NUM_STEPS):
        # reset decision
        for a in range(len(agents)):
            world.setFeature(sides[a], NOT_DECIDED, recurse=True)

        logging.info('====================================')
        logging.info(f'Step {t}')
        step = world.step()
        for a in range(len(agents)):
            logging.info(f'{agents[a].name}: {world.getFeature(sides[a], unique=True)}')
