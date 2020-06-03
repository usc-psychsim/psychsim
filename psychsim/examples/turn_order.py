from psychsim.agent import Agent
from psychsim.helper_functions import multi_set_matrix
from psychsim.pwl import makeTree, CONSTANT, setToFeatureMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of the effect of turn order. There are two agents, one has a \'copy\' action which ' \
                  'simply copies a variable\'s value to another variable, and another agent that has an ' \
                  '\'increment\' action that adds 1 to the variables\'s current value. Different orders between the' \
                  'execution of the agents\'s actions will result in different results for the variables.' \
                  'Note: when agents act in parallel, the copy agent sees the "old" value of the variable.'

AGENT1_ID = 'Agent 1'
AGENT2_ID = 'Agent 2'

if __name__ == '__main__':

    # turn orders
    turn_orders = {'First increment, then copy': [AGENT1_ID, AGENT2_ID],
                   'First copy, then increment': [AGENT2_ID, AGENT1_ID],
                   'Simult. increment and copy': [{AGENT1_ID, AGENT2_ID}]}

    for label, turn_order in turn_orders.items():

        agent1 = Agent(AGENT1_ID)
        agent2 = Agent(AGENT2_ID)

        # create world and add agents
        world = World()
        world.addAgent(agent1)
        world.addAgent(agent2)

        # add variables
        var_counter = world.defineState(agent1.name, 'counter', int, lo=0, hi=3)
        var_copy = world.defineState(agent2.name, 'counter_copy', int, lo=0, hi=3)

        # define first agent's action (counter increment)
        action = agent1.addAction({'verb': '', 'action': 'increment'})
        tree = makeTree(multi_set_matrix(var_counter, {var_counter: 1, CONSTANT: 1}))
        world.setDynamics(var_counter, action, tree)

        # define second agent's action (var is copy from counter)
        action = agent2.addAction({'verb': '', 'action': 'copy'})
        tree = makeTree(setToFeatureMatrix(var_copy, var_counter))
        world.setDynamics(var_copy, action, tree)

        world.setOrder(turn_order)

        # resets vars
        world.setFeature(var_copy, 0)
        world.setFeature(var_counter, 0)

        print('====================================')
        print(label)

        # steps
        for i in range(4):
            print('Step {}, decision by: {}'.format(str(i), turn_order[i % len(turn_order)]))
            step = world.step()
            counter = next(iter(world.getFeature(var_counter).keys()))
            counter_cp = next(iter(world.getFeature(var_copy).keys()))
            print('Counter: {}\tCopy: {}'.format(counter, counter_cp))
            # world.explain(step, level=4) # todo does not work, need outcomes information
