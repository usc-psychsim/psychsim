import logging
from psychsim.agent import Agent
from psychsim.helper_functions import multi_compare_row, set_constant_reward, get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, equalRow, setToConstantMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of using theory-of-mind in a game-theory scenario involving two agents in the Chicken ' \
                  'Game (https://en.wikipedia.org/wiki/Chicken_(game)#Game_theoretic_applications). ' \
                  'Both agents should choose the "go straight" action which is rationally optimal, independently of ' \
                  'the other agent\'s action.'

NUM_STEPS = 3
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# action indexes
NOT_DECIDED = 0
WENT_STRAIGHT = 1
SWERVED = 2

# payoff parameters (according to PD)
SUCKER = -1  # CD
TEMPTATION = 1  # DC
MUTUAL_COOP = 0  # CC
PUNISHMENT = -1000  # DD
INVALID = -10000

DEBUG = False


# defines a payoff matrix tree (0 = didn't decide, 1 = went straight, 2 = swerved)
def get_reward_tree(agent, my_dec, other_dec):
    return makeTree({'if': multi_compare_row({my_dec: 1}, NOT_DECIDED),  # if dec >= 0
                     True: {'if': multi_compare_row({my_dec: -1}, NOT_DECIDED),  # if dec == 0, did not yet decide
                            True: set_constant_reward(agent, INVALID),
                            False: {'if': equalRow(my_dec, SWERVED),  # if dec >=2, I cooperated
                                    True: {'if': equalRow(other_dec, SWERVED),  # if other cooperated
                                           True: set_constant_reward(agent, MUTUAL_COOP),  # both cooperated
                                           False: set_constant_reward(agent, SUCKER)},
                                    False: {'if': equalRow(other_dec, SWERVED),
                                            # if I defected and other cooperated
                                            True: set_constant_reward(agent, TEMPTATION),
                                            False: set_constant_reward(agent, PUNISHMENT)}}},  # both defected
                     False: set_constant_reward(agent, INVALID)})  # invalid


# gets a state description
def get_state_desc(world, dec_feature):
    decision = world.getValue(dec_feature)
    if decision == NOT_DECIDED:
        return 'N/A'
    if decision == WENT_STRAIGHT:
        return 'went straight'
    if decision == SWERVED:
        return 'swerved'


if __name__ == '__main__':

    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create world and add agent
    world = World()
    agent1 = Agent('Agent 1')
    world.addAgent(agent1)
    agent2 = Agent('Agent 2')
    world.addAgent(agent2)

    agents_dec = []
    agents = [agent1, agent2]
    for agent in agents:
        # set agent's params
        agent.setAttribute('discount', 1)
        agent.setAttribute('selection', TIEBREAK)
        agent.setHorizon(1)
        agent.setRecursiveLevel(1)

        # add "decision" variable (0 = didn't decide, 1 = went straight, 2 = swerved)
        dec = world.defineState(agent.name, 'decision', int, lo=0, hi=2)
        world.setFeature(dec, NOT_DECIDED)
        agents_dec.append(dec)

        # define agents' actions (defect and cooperate)
        action = agent.addAction({'verb': '', 'action': 'go straight'})
        tree = makeTree(setToConstantMatrix(dec, WENT_STRAIGHT))
        world.setDynamics(dec, action, tree)
        action = agent.addAction({'verb': '', 'action': 'swerve'})
        tree = makeTree(setToConstantMatrix(dec, SWERVED))
        world.setDynamics(dec, action, tree)

    # defines payoff matrices
    agent1.setReward(get_reward_tree(agent1, agents_dec[0], agents_dec[1]), 1)
    agent2.setReward(get_reward_tree(agent2, agents_dec[1], agents_dec[0]), 1)

    # define order
    my_turn_order = [{agent1.name, agent2.name}]
    world.setOrder(my_turn_order)

    # add true mental model of the other to each agent
    world.setMentalModel(agent1.name, agent2.name, Distribution({get_true_model_name(agent2): 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({get_true_model_name(agent1): 1}))

    for i in range(NUM_STEPS):

        # decision per step (1 per agent): go straight or swerve?
        logging.info('====================================')
        logging.info('Step {}'.format(i))
        step = world.step()
        for j in range(len(agents)):
            logging.info('{0}: {1}'.format(agents[j].name, get_state_desc(world, agents_dec[j])))

        # print('________________________________')
        # world.explain(step, level=2) # todo step does not provide outcomes anymore

        # print('\n') #todo step does not provide outcomes anymore
        # for i in range(len(agents)):
        #     decision_infos = get_decision_info(step, agents[i].name)
        #     explain_decisions(agents[i].name, decision_infos)
