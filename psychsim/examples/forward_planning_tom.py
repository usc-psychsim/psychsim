import logging
from psychsim.agent import Agent
from psychsim.helper_functions import multi_compare_row, set_constant_reward, get_true_model_name
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, equalRow, setToConstantMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of using theory-of-mind in a game-theory scenario involving two agents in the iterated' \
                  'version of the Prisoner\'s dilemma ' \
                  '(https://en.wikipedia.org/wiki/Prisoner%27s_dilemma#The_iterated_prisoner%27s_dilemma)' \
                  'Both agents follow a strategy inspired on tit-for-tat (https://en.wikipedia.org/wiki/Tit_for_tat)' \
                  'Namely, the first action is open and depends on the ' \
                  'beliefs of the agent about the other\'s behavior. From there on, retaliation is applied by always ' \
                  'choosing defect after the first defection of the other agent (non-forgiving).' \
                  'Hence the planning horizon has an influence on the agents\' decision to cooperate or defect:' \
                  '- if horizon is 0, first action for each agent will be random (0 reward), then tit-fot-tat' \
                  '- if horizon is 1, agents will always defect (one-shot decision, other\'s action does not matter' \
                  '- if horizon is 2, first action for each agent will be random, because CC followed by CC = ' \
                  'DC followed by DD = -2, so C or D have the same value independently of the other; then tit-fot-tat' \
                  '- if horizon is >2, agents will always cooperate because they can see each other\'s tit-for-tat ' \
                  'strategy using ToM, and hence believe the other will cooperate if they also cooperate, leading to ' \
                  'highest mutual payoff in the long run.' \
                  'Note: imperfect models can break this belief and make the agents cynical towards each other.'

# parameters
MAX_HORIZON = 4
NUM_STEPS = 4
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# action indexes
NOT_DECIDED = 0
DEFECTED = 1
COOPERATED = 2

# payoff parameters (according to PD)
SUCKER = -3  # CD
TEMPTATION = 0  # DC
MUTUAL_COOP = -1  # CC
PUNISHMENT = -2  # DD
INVALID = -10000

DEBUG = False


# defines a payoff matrix tree (0 = didn't decide, 1 = Defected, 2 = Cooperated)
def get_reward_tree(agent, my_dec, other_dec):
    return makeTree({'if': multi_compare_row({my_dec: 1}, NOT_DECIDED),  # if dec >= 0
                     True: {'if': multi_compare_row({my_dec: -1}, NOT_DECIDED),  # if dec <= 0, did not yet decide
                            True: set_constant_reward(agent, INVALID),
                            False: {'if': equalRow(my_dec, COOPERATED),  # if dec >=2, I cooperated
                                    True: {'if': equalRow(other_dec, COOPERATED),  # if other cooperated
                                           True: set_constant_reward(agent, MUTUAL_COOP),  # both cooperated
                                           False: set_constant_reward(agent, SUCKER)},
                                    False: {'if': equalRow(other_dec, COOPERATED),
                                            # if I defected and other cooperated
                                            True: set_constant_reward(agent, TEMPTATION),
                                            False: set_constant_reward(agent, PUNISHMENT)}}},  # both defected
                     False: set_constant_reward(agent, INVALID)})  # invalid


def get_state_desc(world, dec_feature):
    decision = world.getValue(dec_feature)
    if decision == NOT_DECIDED:
        return 'N/A'
    if decision == DEFECTED:
        return 'defected'
    if decision == COOPERATED:
        return 'cooperated'


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
        agent.setRecursiveLevel(1)

        # add "decision" variable (0 = didn't decide, 1 = Defected, 2 = Cooperated)
        dec = world.defineState(agent.name, 'decision', int, lo=0, hi=2)
        world.setFeature(dec, NOT_DECIDED)
        agents_dec.append(dec)

    # define agents' actions inspired on tit-for-tat: first decision is open, then retaliate non-cooperation.
    # as soon as one agent defects it will always defect from there on
    for i, agent in enumerate(agents):
        my_dec = agents_dec[i]
        other_dec = agents_dec[0 if i == 1 else 1]

        # defect (not legal if other has cooperated before, legal only if agent itself did not defect before)
        action = agent.addAction({'verb': '', 'action': 'defect'},
                                 makeTree({'if': equalRow(other_dec, COOPERATED),
                                           True: {'if': equalRow(my_dec, DEFECTED),
                                                  True: True,
                                                  False: False},
                                           False: True}))
        tree = makeTree(setToConstantMatrix(my_dec, DEFECTED))
        world.setDynamics(my_dec, action, tree)

        # cooperate (not legal if other or agent itself defected before)
        action = agent.addAction({'verb': '', 'action': 'cooperate'},
                                 makeTree({'if': equalRow(other_dec, DEFECTED),
                                           True: False,
                                           False: {'if': equalRow(my_dec, DEFECTED),
                                                   True: False,
                                                   False: True}}))
        tree = makeTree(setToConstantMatrix(my_dec, COOPERATED))
        world.setDynamics(my_dec, action, tree)

    # defines payoff matrices (equal to both agents)
    agent1.setReward(get_reward_tree(agent1, agents_dec[0], agents_dec[1]), 1)
    agent2.setReward(get_reward_tree(agent2, agents_dec[1], agents_dec[0]), 1)

    # define order
    my_turn_order = [{agent1.name, agent2.name}]
    world.setOrder(my_turn_order)

    # add true mental model of the other to each agent
    world.setMentalModel(agent1.name, agent2.name, Distribution({get_true_model_name(agent2): 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({get_true_model_name(agent1): 1}))

    # save log

    for h in range(MAX_HORIZON + 1):
        logging.info('====================================')
        logging.info('Horizon {}'.format(h))

        # set horizon (also to the true model!) and reset decisions
        for i in range(len(agents)):
            agents[i].setHorizon(h)
            agents[i].setHorizon(h, get_true_model_name(agents[i]))
            world.setFeature(agents_dec[i], NOT_DECIDED)

        for t in range(NUM_STEPS):

            # decision per step (1 per agent): cooperate or defect?
            logging.info('---------------------')
            logging.info('Step {}'.format(t))
            step = world.step()
            for i in range(len(agents)):
                logging.info('{0}: {1}'.format(agents[i].name, get_state_desc(world, agents_dec[i])))

            # print('________________________________')
            # world.explain(step, level=2) # todo step does not provide outcomes anymore

            # print('\n') #todo step does not provide outcomes anymore
            # for i in range(len(agents)):
            #     decision_infos = get_decision_info(step, agents[i].name)
            #     explain_decisions(agents[i].name, decision_infos)
