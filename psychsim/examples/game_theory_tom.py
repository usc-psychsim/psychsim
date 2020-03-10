from psychsim.agent import Agent
from psychsim.helper_functions import multi_compare_row, set_constant_reward, get_decision_info, explain_decisions
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, equalRow, setToConstantMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Example of using theory-of-min in a game-theory scenario involving two agents in the Chicken Game' \
                  '(https://en.wikipedia.org/wiki/Chicken_(game)#Game_theoretic_applications). ' \
                  'Both agents should choose the "defect" action which is rationally optimal.'

MODEL_NAME = 'my model'

# parameters (payoffs according to the Chicken Game)
SUCKER = 0  # CD
TEMPTATION = 2  # DC
MUTUAL_COOP = 1  # CC
PUNISHMENT = -10  # DD


# defines a payoff matrix tree (0 = didn't decide, 1 = Defected, 2 = Cooperated)
def get_reward_tree(agent, my_coop, other_coop):
    return makeTree({'if': multi_compare_row({my_coop: 1}, 0),  # if coop >= 0
                     True: {'if': multi_compare_row({my_coop: -1}, 0),  # if coop == 0, did not yet decide
                            True: set_constant_reward(agent, 0),
                            False: {'if': equalRow(my_coop, 2),  # if coop >=2, I cooperated
                                    True: {'if': equalRow(other_coop, 2),  # if other cooperated
                                           True: set_constant_reward(agent, MUTUAL_COOP),  # both cooperated
                                           False: set_constant_reward(agent, SUCKER)},
                                    False: {'if': equalRow(other_coop, 2),  # if I defected and other cooperated
                                            True: set_constant_reward(agent, TEMPTATION),
                                            False: set_constant_reward(agent, PUNISHMENT)}}},  # both defected
                     False: set_constant_reward(agent, 0)})  # invalid


# gets a state description
def get_state_desc(my_world, my_coop):
    result = str(my_world.getFeature(my_coop)).replace("100%\t", "")
    if result == '0':
        return 'start'
    if result == '1':
        return 'defected'
    if result == '2':
        return 'cooperated'


if __name__ == '__main__':

    # create world and add agent
    world = World()
    agent1 = Agent('Agent 1')
    world.addAgent(agent1)
    agent2 = Agent('Agent 2')
    world.addAgent(agent2)

    agents_coop = []
    agents = [agent1, agent2]
    for agent in agents:
        # set agent's params
        agent.setAttribute('discount', 1)
        agent.setHorizon(1)
        agent.setRecursiveLevel(1)

        # add 'cooperated' variable (0 = didn't decide, 1 = Defected, 2 = Cooperated)
        coop = world.defineState(agent.name, 'cooperated', int, lo=0, hi=2)
        world.setFeature(coop, 0)
        agents_coop.append(coop)

        # define agents' actions (defect and cooperate)
        action = agent.addAction({'verb': '', 'action': 'defect'})
        tree = makeTree(setToConstantMatrix(coop, 1))
        world.setDynamics(coop, action, tree)
        action = agent.addAction({'verb': '', 'action': 'cooperate'})
        tree = makeTree(setToConstantMatrix(coop, 2))
        world.setDynamics(coop, action, tree)

        # add mental model
        agent.addModel(MODEL_NAME, parent='{}0'.format(agent.name))

    # defines payoff matrices
    agent1.setReward(get_reward_tree(agent1, agents_coop[0], agents_coop[1]), 1)
    agent2.setReward(get_reward_tree(agent2, agents_coop[1], agents_coop[0]), 1)

    # define order
    my_turn_order = [{agent1.name, agent2.name}]
    world.setOrder(my_turn_order)

    # # add mental models
    # agent1.addModel(MODEL_NAME, parent='{}0'.format(agent1.name))
    # agent2.addModel(MODEL_NAME, parent='{}0'.format(agent2.name))

    # add mental model of the other for each agent
    world.setMentalModel(agent1.name, agent2.name, Distribution({MODEL_NAME: 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({MODEL_NAME: 1}))

    # *single* decision (1 per agent): cooperate or defect?
    for i in range(len(my_turn_order)):
        print('====================================')
        print('Step {}'.format(i))
        step = world.step()
        for j in range(len(agents)):
            print(agents[j].name + ': ' + get_state_desc(world, agents_coop[j]))

        # print('________________________________')
        # world.explain(step, level=2) # todo step does not provide outcomes anymore

        # print('\n') todo step does not provide outcomes anymore
        # for i in range(len(agents)):
        #     decision_infos = get_decision_info(step, agents[i].name)
        #     explain_decisions(agents[i].name, decision_infos)
