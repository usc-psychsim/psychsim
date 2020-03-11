from psychsim.agent import Agent
from psychsim.helper_functions import multi_compare_row, set_constant_reward, get_true_model_name, \
    get_decision_info, explain_decisions, get_feature_values
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, equalRow, setToConstantMatrix, equalFeatureRow
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Example of using theory-of-min in a game-theory scenario involving two agents in the Chicken Game' \
                  '(https://en.wikipedia.org/wiki/Chicken_(game)#Game_theoretic_applications). ' \
                  'Both agents should choose the "defect" action which is rationally optimal.'

NUM_STEPS = 4
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# # action indexes
# NOT_DECIDED = 0
# STRAIGHT = 1
# SWERVE_RIGHT = 2

# # parameters (payoffs according to the Chicken Game)
# CHICKEN = -1  # RS
# DARE = 1  # SR
# MUTUAL_COOP = 0  # RR
# PUNISHMENT = -10  # SS
# INVALID = -10000

# action indexes
NOT_DECIDED = 0
DEFECTED = 1
COOPERATED = 2

# payoff parameters (according to PD)
SUCKER = -5  # CD
TEMPTATION = 0  # DC
MUTUAL_COOP = -1  # CC
PUNISHMENT = -3  # DD
INVALID = -10000


# # defines a payoff matrix tree (0 = didn't decide, 1 = go straight, 2 = swerve right)
# def get_reward_tree(agent, my_dec, other_dec):
#     return makeTree({'if': multi_compare_row({my_dec: 1}, NOT_DECIDED),  # if dec >= 0
#                      True: {'if': multi_compare_row({my_dec: -1}, NOT_DECIDED),  # if dec == 0, did not yet decide
#                             True: set_constant_reward(agent, INVALID),
#                             False: {'if': equalRow(my_dec, SWERVE_RIGHT),  # if dec >=2, I cooperated
#                                     True: {'if': equalRow(other_dec, SWERVE_RIGHT),  # if other cooperated
#                                            True: set_constant_reward(agent, MUTUAL_COOP),  # both cooperated
#                                            False: set_constant_reward(agent, CHICKEN)},
#                                     False: {'if': equalRow(other_dec, SWERVE_RIGHT),
#                                             # if I defected and other cooperated
#                                             True: set_constant_reward(agent, DARE),
#                                             False: set_constant_reward(agent, PUNISHMENT)}}},  # both defected
#                      False: set_constant_reward(agent, INVALID)})  # invalid

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


# # gets a state description
# def get_state_desc(world, dec_feature):
#     decision = get_feature_values(world.getFeature(dec_feature))[0][0]
#     if decision == NOT_DECIDED:
#         return 'N/A'
#     if decision == STRAIGHT:
#         return 'went straight'
#     if decision == SWERVE_RIGHT:
#         return 'swerved right'

# gets a state description
def get_state_desc(world, dec_feature):
    decision = get_feature_values(world.getFeature(dec_feature))[0][0]
    if decision == NOT_DECIDED:
        return 'N/A'
    if decision == DEFECTED:
        return 'defected'
    if decision == COOPERATED:
        return 'cooperated'


if __name__ == '__main__':

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
        agent.setHorizon(2)
        agent.setRecursiveLevel(1)

        # add "decision" variable (0 = didn't decide, 1 = Defected, 2 = Cooperated)
        dec = world.defineState(agent.name, 'decision', int, lo=0, hi=2)
        world.setFeature(dec, NOT_DECIDED)
        agents_dec.append(dec)

    # define agents' actions:
    for i, agent in enumerate(agents):
        dec = agents_dec[i]

        # defect is always available
        action = agent.addAction({'verb': '', 'action': 'defect'})
        tree = makeTree(setToConstantMatrix(dec, DEFECTED))
        world.setDynamics(dec, action, tree)

        # cooperate based on tit-for-tat: only available if agent did not decide yet or cooperated before,
        # otherwise agent will always defect
        other_ag_idx = 0 if i == 1 else 1
        action = agent.addAction({'verb': '', 'action': 'cooperate'},
                                 makeTree({'if': equalRow(agents_dec[other_ag_idx], DEFECTED),
                                           True: False,
                                           False: True}))
        tree = makeTree(setToConstantMatrix(dec, COOPERATED))
        world.setDynamics(dec, action, tree)

    # defines payoff matrices (equal to both agents)
    agent1.setReward(get_reward_tree(agent1, agents_dec[0], agents_dec[1]), 1)
    agent2.setReward(get_reward_tree(agent2, agents_dec[1], agents_dec[0]), 1)

    # define order
    my_turn_order = [{agent1.name, agent2.name}]
    world.setOrder(my_turn_order)

    # add true mental model of the other to each agent
    world.setMentalModel(agent1.name, agent2.name, Distribution({get_true_model_name(agent2): 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({get_true_model_name(agent1): 1}))

    for i in range(NUM_STEPS):

        # decision per step (1 per agent): cooperate or defect?
        print('====================================')
        print('Step {}'.format(i))
        step = world.step(tiebreak=TIEBREAK)
        for j in range(len(agents)):
            print('{0}: {1}'.format(agents[j].name, get_state_desc(world, agents_dec[j])))

        # print('________________________________')
        # world.explain(step, level=2) # todo step does not provide outcomes anymore

        # print('\n') #todo step does not provide outcomes anymore
        # for i in range(len(agents)):
        #     decision_infos = get_decision_info(step, agents[i].name)
        #     explain_decisions(agents[i].name, decision_infos)
