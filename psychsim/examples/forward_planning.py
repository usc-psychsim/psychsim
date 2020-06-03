from psychsim.agent import Agent
from psychsim.world import World
from psychsim.helper_functions import multi_set_matrix, get_decision_info, explain_decisions
from psychsim.pwl import makeTree, setToFeatureMatrix
from psychsim.reward import achieveFeatureValue, CONSTANT

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Simple forward planning (discounted) example involving a single decision and variable rewards.' \
                  'Agent will prefer to go left unless it can plan 3+ steps into the future, in which case ' \
                  'it "sees" it\'s best to go right to later achieve a higher reward.'

# parameters
MAX_HORIZON = 5
DISCOUNT = 0.9

if __name__ == '__main__':
    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # set discount
    agent.setAttribute('discount', DISCOUNT)

    # add position variable
    pos = world.defineState(agent.name, 'position', int, -100, 100, 'Agent\'s location')

    # define agents' actions (stay, left and right)
    action = agent.addAction({'verb': 'move', 'action': 'nowhere'})
    tree = makeTree(setToFeatureMatrix(pos, pos))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'left'})
    tree = makeTree(multi_set_matrix(pos, {pos: 1, CONSTANT: -1}))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(multi_set_matrix(pos, {pos: 1, CONSTANT: 1}))
    world.setDynamics(pos, action, tree)

    # define rewards (left always adds 1, right depends on position)
    agent.setReward(achieveFeatureValue(pos, -1, agent.name), 1)
    agent.setReward(achieveFeatureValue(pos, -2, agent.name), 2)
    agent.setReward(achieveFeatureValue(pos, -3, agent.name), 3)
    agent.setReward(achieveFeatureValue(pos, 2, agent.name), 3)
    agent.setReward(achieveFeatureValue(pos, 3, agent.name), 100)

    world.setOrder([{agent.name}])

    for i in range(MAX_HORIZON + 1):
        print('====================================')
        print('Horizon: {}'.format(str(i)))

        # reset
        world.setFeature(pos, 0)
        agent.setHorizon(i)

        # single decision: left or right?
        step = world.step()
        # print(step)
        print('Position: {}'.format(world.getValue(pos)))
        # world.explain(step, level=3) # todo not working, cannot retrieve old 'outcomes' from step

        # print('\n')
        # decision_infos = get_decision_info(step, agent.name) # todo cannot retrieve old 'outcomes' from step
        # explain_decisions(agent.name, decision_infos)
