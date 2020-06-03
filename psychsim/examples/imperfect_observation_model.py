import logging
from psychsim.agent import Agent
from psychsim.helper_functions import multi_set_matrix, multi_reward_matrix
from psychsim.pwl import makeTree, equalRow, setToFeatureMatrix, setToConstantMatrix, thresholdRow, CONSTANT
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of setting a incorrect belief over another agent\'s feature. Two agents interact with each ' \
                  'other: a a consumer agent asks for a certain amount of product while a producer agent produces ' \
                  'that product. The produced amount depends on the producer\'s production capacity and the asked ' \
                  'amount, i.e., produced=asked*capacity. The consumer asks for product according to the producer\'s ' \
                  'capacity: if it observes that the capacity is "full", it will order a "normal" amount, otherwise ' \
                  'it will ask for double the "normal" amount if it observes that the capacity is "half". ' \
                  'The consumer\'s reward is dictated by the amount of product received: if it receives the normal ' \
                  'amount it receives a positive reward, if it receives more than the normal amount it will receive a ' \
                  'penalty to simulate an over-stock cost.' \
                  'We set a belief to the consumer agent so that it always believes the consumer is producing at half' \
                  'capacity when in reality it is not. The agent will therefore always ask for more, thinking that ' \
                  'this way it will receive what it really needs. This simulates hoarding behavior under false beliefs.'

# parameters
NUM_STEPS = 4
HORIZON = 2

DEBUG = False


def get_fake_model_name(agent):
    return 'fake {} model'.format(agent.name)


if __name__ == '__main__':

    # sets up log to screen
    logging.basicConfig(format='%(message)s', level=logging.DEBUG if DEBUG else logging.INFO)

    # create world and add agents
    world = World()
    ag_producer = Agent('Producer')
    world.addAgent(ag_producer)
    ag_consumer = Agent('Consumer')
    world.addAgent(ag_consumer)
    agents = [ag_producer, ag_consumer]

    # agent settings
    ag_producer.setAttribute('discount', 1)
    ag_producer.setHorizon(HORIZON)
    ag_consumer.setAttribute('discount', 1)
    ag_consumer.setHorizon(HORIZON)

    # add variables (capacity and asked/received amounts)
    var_half_cap = world.defineState(ag_producer.name, 'half capacity', bool)
    world.setFeature(var_half_cap, False)
    var_ask_amnt = world.defineState(ag_producer.name, 'asked amount', int, lo=0, hi=100)
    world.setFeature(var_ask_amnt, 0)
    var_rcv_amnt = world.defineState(ag_consumer.name, 'received amount', int, lo=0, hi=100)
    world.setFeature(var_rcv_amnt, 0)

    # add producer actions
    # produce capacity: if half capacity then 0.5*asked amount else asked amount)
    act_prod = ag_producer.addAction({'verb': '', 'action': 'produce'})
    tree = makeTree({'if': equalRow(var_half_cap, True),
                     True: multi_set_matrix(var_rcv_amnt, {var_ask_amnt: 0.5}),
                     False: setToFeatureMatrix(var_rcv_amnt, var_ask_amnt)})
    world.setDynamics(var_rcv_amnt, act_prod, tree)

    # add consumer actions (ask more = 10 / less = 5)
    act_ask_more = ag_consumer.addAction({'verb': '', 'action': 'ask_more'})
    tree = makeTree(setToConstantMatrix(var_ask_amnt, 10))
    world.setDynamics(var_ask_amnt, act_ask_more, tree)

    act_ask_less = ag_consumer.addAction({'verb': '', 'action': 'ask_less'})
    tree = makeTree(setToConstantMatrix(var_ask_amnt, 5))
    world.setDynamics(var_ask_amnt, act_ask_less, tree)

    # defines payoff for consumer agent: if received amount > 5 then 10 - rcv_amnt (penalty) else rcv_amount (reward)
    # this simulates over-stock cost, best is to receive max of 5, more than this has costs
    ag_consumer.setReward(
        makeTree({'if': thresholdRow(var_rcv_amnt, 5),
                  True: multi_reward_matrix(ag_consumer, {CONSTANT: 10, var_rcv_amnt: -1}),
                  False: multi_reward_matrix(ag_consumer, {var_rcv_amnt: 1})}),
        1)

    # define order (parallel execution)
    world.setOrder([{ag_producer.name, ag_consumer.name}])

    # sets consumer belief that producer is at half-capacity, making it believe that asking more has more advantage
    # - in reality, producer is always at full capacity, so best strategy would be to always ask less
    ag_consumer.setBelief(var_half_cap, True)

    total_rwd = 0
    for i in range(NUM_STEPS):
        logging.info('====================================')
        logging.info('Step {}'.format(i))
        step = world.step()
        reward = ag_consumer.reward()
        logging.info('Half capacity:\t\t{}'.format(world.getValue(var_half_cap)))
        logging.info('Asked amount:\t\t{}'.format(world.getValue(var_ask_amnt)))
        logging.info('Received amount:\t{}'.format(world.getValue(var_rcv_amnt)))
        logging.info('Consumer reward:\t{}'.format(reward))
        total_rwd += reward

        logging.info('________________________________')
        # world.explain(step, level=2)# todo step does not provide outcomes anymore

    logging.info('====================================')
    logging.info('Total reward: {0}'.format(total_rwd))
