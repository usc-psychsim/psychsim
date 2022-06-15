from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import makeTree, setToFeatureMatrix, incrementMatrix, modelKey
from psychsim.reward import maximizeFeature
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example that models one agent navigating left or right along one dimension (one position feature).' \
                  ' We set a belief to the agent for an incorrect position, and see the agent act based on that ' \
                  'belief, i.e., in its mind the position is changing, but it is not aligned with the true position.'

# parameters
HORIZON = 1
DISCOUNT = 1
MAX_STEPS = 3


def _get_belief(feature: str, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(agent.name, unique=True)
    return world.getFeature(feature, state=agent.getBelief(model=model))


if __name__ == '__main__':

    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # set parameters
    agent.setAttribute('discount', DISCOUNT)
    agent.setHorizon(HORIZON)

    # add position variable
    pos = world.defineState(agent.name, 'position', int, lo=-100, hi=100)
    world.setFeature(pos, 0)

    # define agents' actions (stay 0, left -1 and right +1)
    action = agent.addAction({'verb': 'move', 'action': 'nowhere'})
    tree = makeTree(setToFeatureMatrix(pos, pos))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'left'})
    tree = makeTree(incrementMatrix(pos, -1))
    world.setDynamics(pos, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(incrementMatrix(pos, 1))
    world.setDynamics(pos, action, tree)

    # define rewards (maximize position, i.e., always go right)
    agent.setReward(maximizeFeature(pos, agent.name), 1)

    # set order
    world.setOrder([agent.name])

    # agent has initial belief about its position, which will be updated after executing actions
    agent.set_observations(unobservable=[pos])  # commenting this makes agent observe true pos after 1st step
    # agent.setBelief(pos, 10, agent.get_true_model())  # deterministic belief
    agent.setBelief(pos, Distribution({10: 0.5, 12: 0.5}))  # stochastic belief

    print('====================================')
    print(f'Initial belief about pos:\n{_get_belief(pos)}')

    for i in range(MAX_STEPS):
        print('====================================')
        print(f'Current pos: {world.getFeature(pos)}')

        # decision: left, right or no-move?
        step = world.step()

        # prints all models and beliefs
        print('____________________________________')
        print(f'Updated belief about pos:\n{_get_belief(pos)}')
        print('____________________________________')
