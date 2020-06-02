from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, setToConstantMatrix
from psychsim.helper_functions import multi_compare_row, set_constant_reward, set_illegal_action, set_legal_action, \
    get_true_model_name

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of how to set legality for agent\'s models in PsychSim using the helper functions. ' \
                  'There are two agents how have two actions: go left or go right. If the agents choose the same ' \
                  'action, they both receive a penalty, if they go to different directions they both receive a ' \
                  'reward. We create mental models of each agent and then add them to the other. Then we "remove" the' \
                  'right action of the *models* be setting it always illegal using the helper functions. This results' \
                  'in the agents always choosing to go right, because during ToM reasoning they believe that going ' \
                  'left is the only option available to the other.'

# parameters (positive reward if sides are different, otherwise punishment)
DIFF_SIDES_RWD = 1
SAME_SIDE_RWD = -1

NUM_STEPS = 4
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# action indexes
NOT_DECIDED = 0
WENT_LEFT = 1
WENT_RIGHT = 2


def get_fake_model_name(agent):
    return 'fake {} model'.format(agent.name)


# defines reward tree
def get_reward_tree(agent, my_side, other_side):
    return makeTree({'if': multi_compare_row({my_side: 1}),  # if my_side >= 0
                     True: {'if': multi_compare_row({my_side: -1}),  # if my_side == 0, did not yet decide
                            True: set_constant_reward(agent, 0),
                            False: {'if': multi_compare_row({my_side: 1, other_side: -1}),  # if my_side >= other_side
                                    True: {'if': multi_compare_row({other_side: 1, my_side: -1}),
                                           True: set_constant_reward(agent, SAME_SIDE_RWD),
                                           False: set_constant_reward(agent, DIFF_SIDES_RWD)},
                                    False: set_constant_reward(agent, DIFF_SIDES_RWD)}},
                     False: set_constant_reward(agent, 0)})


# gets a state description
def get_state_desc(world, side):
    result = world.getValue(side)
    if result == NOT_DECIDED:
        return 'N/A'
    if result == WENT_LEFT:
        return 'went left'
    if result == WENT_RIGHT:
        return 'went right'


if __name__ == '__main__':

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
        side = world.defineState(agent.name, 'side', int, lo=0, hi=2)
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

        # create a mental model of the agent
        agent.addModel(get_fake_model_name(agent), parent=get_true_model_name(agent))

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

    # ** unnecessary / just for illustration **: set left actions legal for both the agents and their models
    set_legal_action(agent1, lefts[0], [get_true_model_name(agent1), get_fake_model_name(agent1)])
    set_legal_action(agent2, lefts[1], [get_true_model_name(agent2), get_fake_model_name(agent2)])

    for i in range(NUM_STEPS):
        # reset decision
        for j in range(len(agents)):
            world.setFeature(sides[j], NOT_DECIDED)

        print('====================================')
        print('Step {0}'.format(str(i)))
        step = world.step()
        for j in range(len(agents)):
            print('{0}: {1}'.format(agents[j].name, get_state_desc(world, sides[j])))

        print('________________________________')
        # world.explain(step, level=2) # todo step does not provide outcomes anymore
