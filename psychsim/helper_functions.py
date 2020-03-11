from psychsim.agent import Agent
from psychsim.probability import Distribution
from psychsim.pwl import KeyedMatrix, KeyedVector, makeFuture, KeyedPlane, setToConstantMatrix, rewardKey

__author__ = 'Pedro Sequeira, Stacy Marsella'
__email__ = 'pedro.sequeira@sri.com'


def multi_set_matrix(key, scaled_keys):
    """
    Performs a linear combination of the given keys, i.e., scales and sums all the keys in scaled_keys dict and adds
    offset if CONSTANT in scaled_keys. If the key itself is in scaled_keys, it adds to its scaled value.
    :param str key: the named key.
    :param dict scaled_keys: the dictionary containing the weights (scalars) for each named key.
    :rtype: KeyedMatrix
    :return: a matrix performing the given linear combination to the given key.
    """
    return KeyedMatrix({makeFuture(key): KeyedVector(scaled_keys)})


def multi_compare_row(scaled_keys, threshold=0.0):
    """
    Gets a hyperplane that performs a comparison between a threshold and a linear combination of named keys.
    :param dict scaled_keys: the dictionary containing the weights (scalars) for each named key.
    :param float threshold: the comparison threshold to the linear combination.
    :rtype: KeyedPlane
    :return: a hyperplane that performs a comparison between the given linear combination and threshold.
    """
    return KeyedPlane(KeyedVector(scaled_keys), threshold)


def set_constant_reward(agent, value):
    """
    Gets a matrix that sets the reward the reward of the given agent to a constant value.
    :param Agent agent: the agent we want to set the reward.
    :param float value: the value we want to set the reward to.
    :rtype: KeyedMatrix
    :return: a matrix that allows setting the agent's reward to the given constant value.
    """
    return setToConstantMatrix(rewardKey(agent.name), value)


def get_true_model_name(agent):
    """
    Gets the name of the "True" model associated with the given agent.
    Node: this depends on the internals of PsychSim, so this function should probably be moved inside Agent.py.
    :param Agent agent: the agent whose true model we want to retrieve.
    :rtype: str
    :return: the name of true model of the given agent.
    """
    return '{}0'.format(agent.name)


def get_feature_values(feature):
    """
    Gets all the values associated with the given feature and corresponding probabilities.
    :param Distribution feature: the state feature, i.e., a probability distribution over values.
    :rtype list[tuple]
    :return: a list containing tuples in the form (value, probability) for each value associated with the given feature.
    """
    return list(zip(feature._domain.values(), feature.values()))


class DecisionInfo(object):
    """
    Corresponds to the summarized information about a decision.
    """

    def __init__(self):
        self.value = .0
        self.reward = .0
        self.horizons_left = 0
        self.actions = {}
        self.state = {}


def get_decision_info(outcomes, decision_agent_name):
    # checks no decisions, return empty list
    if len(outcomes['decisions']) == 0 or decision_agent_name not in outcomes['decisions']:
        return []

    # collects decision info for the agent
    decision = outcomes['decisions'][decision_agent_name]
    dec_info = DecisionInfo()

    # gets planned action (what agent decided to do)
    action = decision['action']

    # check if agent does not have values (only had one action available)
    if 'V' not in decision:
        # return info about the action
        dec_info.actions[decision_agent_name] = action
        for feat_name, feat_value in outcomes['new'].items():
            dec_info.state[feat_name] = feat_value
        return [dec_info]

    # otherwise, get info on the optimal action by the agent
    action_info = decision['V'][action]

    # gets planning decisions
    projection = [action_info[key] for key in action_info.keys() if key != '__EV__'][0]['projection']

    # checks for no projection (no planning)
    if len(projection) == 0:
        return []

    projection = projection[0]

    # collects actions planned for all agents
    if 'actions' in projection:
        for ag_name, ag_action in projection['actions'].items():
            dec_info.actions[ag_name] = str(next(iter(ag_action)))

    # collects state at this horizon
    for feat_name, feat_value in projection['state'].items():
        dec_info.state[feat_name] = feat_value

    # collects other info
    dec_info.value = float(projection['V'])
    dec_info.reward = float(projection['R'])
    dec_info.horizons_left = int(projection['horizon'])

    # adds next planning horizon
    decision_infos = get_decision_info(projection['projection'], decision_agent_name)
    decision_infos.insert(0, dec_info)
    return decision_infos


IGNORE_FEATURES = {'_model', '_turn'}


def explain_decisions(ag_name, decision_infos):
    print('=============================================')
    print('{}\'s planning ({} steps)'.format(ag_name, len(decision_infos)))

    for i in range(len(decision_infos)):
        dec_info = decision_infos[i]

        print('\t---------------------------------------------')
        print('\tPlan horizon {} ({} left):'.format(i, dec_info.horizons_left))

        print('\t\tProjected agents\' actions:')
        for ag_act_name, ag_action in dec_info.actions.items():
            print('\t\t\t{}: {}'.format(ag_act_name, ag_action))
        print('\t\t---------------------------------------------')
        print('\t\tProjected resulting state:')
        for feat_name, feat_val in dec_info.state.items():
            if feat_name == '' or any(ignore in feat_name for ignore in IGNORE_FEATURES):
                continue
            print('\t\t\t{}: {}'.format(feat_name, feat_val))
        print('\t\t---------------------------------------------')
        print('\t\tReward received (by {}): {}'.format(ag_name, dec_info.reward))
        print('\t\tValue (for {}): {}'.format(ag_name, dec_info.value))
