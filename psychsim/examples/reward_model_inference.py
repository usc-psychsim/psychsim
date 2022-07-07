import random
from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.world import World
from psychsim.pwl import makeTree, modelKey, incrementMatrix, setToConstantMatrix, rewardKey
from psychsim.reward import maximizeFeature, minimizeFeature

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'
__description__ = 'Example of inference over reward models of another agent based on observations of their behavior.' \
                  'One agent (the actor) moves in the environment (always goes right).' \
                  'Another agent (the observer) maintains a belief over the actor\'s reward model (either an agent ' \
                  'that prefers going right, going left or acting randomly. This belief is updated based on ' \
                  'observations of the actor\'s actions over time.'

# parameters
MAX_STEPS = 100
AGENT_NAME = 'actor'
OBSERVER_NAME = 'observer'
HORIZON = 2
AGENT_SELECTION = 'random'
SEED = 173


def _get_belief(feature: str, ag: Agent, model: str = None) -> Distribution:
    if model is None:
        model = world.getModel(ag.name, unique=True)
    return world.getFeature(feature, state=ag.getBelief(model=model))


if __name__ == '__main__':
    random.seed(SEED)

    # create world and add actor agent
    world = World()
    agent = world.addAgent(AGENT_NAME)
    agent.setAttribute('horizon', HORIZON)
    agent.setAttribute('selection', AGENT_SELECTION)

    # create observer (no actions)
    observer = world.addAgent(OBSERVER_NAME)

    # add location variable
    loc = world.defineState(agent.name, 'location', int, -1000, 1000, 'Agent\'s location')
    world.setFeature(loc, 0)

    # define agents' actions (left and right)
    action = agent.addAction({'verb': 'move', 'action': 'left'})
    tree = makeTree(incrementMatrix(loc, -1))
    world.setDynamics(loc, action, tree)
    action = agent.addAction({'verb': 'move', 'action': 'right'})
    tree = makeTree(incrementMatrix(loc, 1))
    world.setDynamics(loc, action, tree)

    # define true reward (maximize loc)
    agent.setReward(maximizeFeature(loc, agent.name), 1)

    world.setOrder([{agent.name}])

    # add agent models (prefer positive vs negative location vs random agent)
    true_model = agent.get_true_model()
    prefer_pos_model = 'prefer_positive_loc'
    agent.addModel(prefer_pos_model, parent=true_model)
    agent.setReward(maximizeFeature(loc, agent.name), 1., model=prefer_pos_model)

    prefer_neg_model = 'prefer_negative_loc'
    agent.addModel(prefer_neg_model, parent=true_model)
    agent.setReward(minimizeFeature(loc, agent.name), 1., model=prefer_neg_model)

    prefer_nothing = 'prefer_nothing'  # random agent
    agent.addModel(prefer_nothing, parent=true_model)
    agent.setReward(setToConstantMatrix(rewardKey(agent.name), 0.), 1., model=prefer_nothing)

    # null_model = agent.zero_level(sample=True)  # TODO this seems to lead to a leak into the agent's true model?

    # set uniform belief over agent's model in observer
    model_names = [name for name in agent.models.keys() if name != true_model]
    dist = Distribution({model: 1. / (len(agent.models) - 1) for model in model_names})
    world.setMentalModel(observer.name, agent.name, dist)

    # agent models ignore the observer
    agent.ignore(observer.name)
    for model in model_names:
        agent.setAttribute('beliefs', True, model=model)
        agent.ignore(observer.name, model=model)

    # observer does not observe agent's true model
    observer.set_observations()

    agent_model = modelKey(agent.name)
    print('====================================')
    print(f'Initial loc: {world.getFeature(loc)}')
    print(f'Initial belief about agent\'s model:\n{_get_belief(agent_model, observer)}')

    for i in range(MAX_STEPS):
        print('====================================')
        step = world.step()
        print(f'Current loc: {world.getFeature(loc)}')
        print(f'Updated belief about agent\'s model:\n{_get_belief(agent_model, observer)}')
