import logging
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, equalRow, setToConstantMatrix, rewardKey
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of using theory-of-mind in a game-theory scenario involving two agents in the Chicken ' \
                  'Game (https://en.wikipedia.org/wiki/Chicken_(game)#Game_theoretic_applications). ' \
                  'Both agents should choose the "go straight" action which is rationally optimal, independently of ' \
                  'the other agent\'s action.'

NUM_STEPS = 3
TIEBREAK = 'random'  # when values of decisions are the same, choose randomly

# decision labels
NOT_DECIDED = 'none'
WENT_STRAIGHT = 'straight'
SWERVED = 'swerved'

# payoff parameters (according to PD)
SUCKER = -1  # CD
TEMPTATION = 1  # DC
MUTUAL_COOP = 0  # CC
PUNISHMENT = -1000  # DD
INVALID = -10000

DEBUG = False


# defines a payoff matrix tree (0 = didn't decide, 1 = went straight, 2 = swerved)
def get_reward_tree(agent, my_dec, other_dec):
    reward_key = rewardKey(agent.name)
    return makeTree({'if': equalRow(my_dec, NOT_DECIDED),  # if I have not decided
                     True: setToConstantMatrix(reward_key, INVALID),
                     False: {'if': equalRow(other_dec, NOT_DECIDED),  # if other has not decided
                             True: setToConstantMatrix(reward_key, INVALID),
                             False: {'if': equalRow(my_dec, SWERVED),  # if I cooperated
                                     True: {'if': equalRow(other_dec, SWERVED),  # if other cooperated
                                            True: setToConstantMatrix(reward_key, MUTUAL_COOP),  # both cooperated
                                            False: setToConstantMatrix(reward_key, SUCKER)},
                                     False: {'if': equalRow(other_dec, SWERVED),
                                             # if I defected and other cooperated
                                             True: setToConstantMatrix(reward_key, TEMPTATION),
                                             False: setToConstantMatrix(reward_key, PUNISHMENT)}}}})  # both defected


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
        # agent.setRecursiveLevel(1)

        # add "decision" variable (0 = didn't decide, 1 = went straight, 2 = swerved)
        dec = world.defineState(agent.name, 'decision', list, [NOT_DECIDED, WENT_STRAIGHT, SWERVED])
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
    world.setMentalModel(agent1.name, agent2.name, Distribution({agent2.get_true_model(): 1}))
    world.setMentalModel(agent2.name, agent1.name, Distribution({agent1.get_true_model(): 1}))

    for i in range(NUM_STEPS):

        # decision per step (1 per agent): go straight or swerve?
        logging.info('====================================')
        logging.info(f'Step {i}')
        step = world.step()
        for a in range(len(agents)):
            logging.info(f'{agents[a].name}: {world.getFeature(agents_dec[a], unique=True)}')
