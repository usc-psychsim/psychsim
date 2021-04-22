from psychsim.agent import Agent
from psychsim.world import World
from psychsim.pwl import makeTree, setToFeatureMatrix, dynamicsMatrix, rewardKey
from psychsim.reward import achieveFeatureValue, CONSTANT

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Simple forward planning (discounted) example involving a single decision and variable rewards.' \
                  'Agent will prefer to go left unless it can plan 3+ steps into the future, in which case ' \
                  'it "sees" it\'s best to go right to later achieve a higher reward.'

# parameters
MAX_HORIZON = 5
MAX_STEPS = 3
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
    tree = makeTree(dynamicsMatrix(pos, {pos: 1, CONSTANT: -1}))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(dynamicsMatrix(pos, {pos: 1, CONSTANT: 1}))
    world.setDynamics(pos, action, tree)

    # define rewards (left always adds 1, right depends on position)
    agent.setReward(achieveFeatureValue(pos, -1, agent.name), 1)
    agent.setReward(achieveFeatureValue(pos, -2, agent.name), 2)
    agent.setReward(achieveFeatureValue(pos, -3, agent.name), 3)
    agent.setReward(achieveFeatureValue(pos, 2, agent.name), 3)
    agent.setReward(achieveFeatureValue(pos, 3, agent.name), 100)

    world.setOrder([{agent.name}])

    true_model = agent.get_true_model()
    for h in range(MAX_HORIZON + 1):
        print('====================================')
        print('Horizon: {}'.format(str(h)))

        # reset
        world.setFeature(pos, 0)
        agent.setHorizon(h)

        for t in range(MAX_STEPS):
            print('____________________________________')
            print('Step: {}'.format(str(t)))

            # left or right?
            debug = {agent.name: {}}
            world.step(debug=debug)
            action = debug[agent.name]['__decision__'][true_model]['action']
            rwd = debug[agent.name]['__decision__'][true_model]['V'][action]['__ER__']
            rwd = None if len(rwd) == 0 else rwd[0]
            print('Position: {}'.format(world.getFeature(pos, unique=True)))
            print('Reward from decision: {}'.format(rwd))
            print('Reward from state: {}'.format(world.getFeature(rewardKey(agent.name), unique=True)))
