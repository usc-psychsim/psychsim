from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import rewardKey, setToConstantMatrix, makeTree, setToFeatureMatrix
from psychsim.world import World

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of modeling local observability, i.e., only observing some feature when some conditions ' \
                  'are met. Here, one agent can open a box to see whether there is a ball inside. The agent does ' \
                  'not know (directly observe) whether the ball is inside the box but has a "seen" feature ' \
                  'that is set to the truth value of whether the ball is in the box once the agent opens the box.'


def _get_belief(feature: str, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(agent.name, unique=True)
    return world.getFeature(feature, state=agent.getBelief(model=model))


if __name__ == '__main__':
    # create world and add agent
    world = World()
    agent = Agent('Agent')
    world.addAgent(agent)

    # add unobservable feature
    ball_box = world.defineState(agent.name, 'Ball in Box', bool)
    world.setFeature(ball_box, True)

    # add observation feature (conditioned on box state)
    see_ball = world.defineState(agent.name, 'See Ball', bool)
    world.setFeature(see_ball, False)

    # define agent's single action, which reveals the ball (if inside the box)
    action = agent.addAction({'verb': 'open box'})
    tree = makeTree(setToFeatureMatrix(see_ball, ball_box))
    world.setDynamics(see_ball, action, tree)

    agent.setReward(setToConstantMatrix(rewardKey(agent), 0.))  # define 0 reward
    world.setOrder([agent.name])  # set order

    # agent has "ball seen" observation, which could be used in the dynamics for other features or reward
    agent.set_observations(unobservable=[ball_box])  # not necessary
    agent.resetBelief()  # not necessary, just for testing

    print('\nBefore step():')
    print(f'Ball in box: {world.getFeature(ball_box)}')
    print(f'Agent\'s belief about ball in box: {_get_belief(ball_box)}')
    print(f'Agent sees ball: {world.getFeature(see_ball)}')
    step = world.step()

    print('\nAfter step():')
    print(f'Ball in box: {world.getFeature(ball_box)}')
    print(f'Agent\'s belief about ball in box: {_get_belief(ball_box)}')
    print(f'Agent sees ball: {world.getFeature(see_ball)}')
