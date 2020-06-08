import random
import numpy as np
from psychsim.world import World
from psychsim.agent import Agent
from psychsim.action import ActionSet
from psychsim.probability import Distribution
from psychsim.pwl import KeyedMatrix, KeyedVector, makeFuture, KeyedPlane, setToConstantMatrix, rewardKey, modelKey, \
    equalRow, makeTree

__author__ = 'Pedro Sequeira, Stacy Marsella'
__email__ = 'pedrodbs@gmail.com'


"""
    PWL UTILITIES
"""


def multi_set_matrix(key, scaled_keys):
    """
    Performs a linear combination of the given keys, i.e., scales and sums all the keys in scaled_keys dict and adds
    offset if CONSTANT in scaled_keys. If the key itself is in scaled_keys, it adds to its scaled value.
    Sets the result to the given feature's future value.
    :param str key: the named key.
    :param dict scaled_keys: the dictionary containing the weights (scalars) for each named key.
    :rtype: KeyedMatrix
    :return: a matrix performing the given linear combination to the given key.
    """
    return KeyedMatrix({makeFuture(key): KeyedVector(scaled_keys)})


def multi_reward_matrix(agent, scaled_keys):
    """
    Performs a linear combination of the given keys, i.e., scales and sums all the keys in scaled_keys dict and adds
    offset if CONSTANT in scaled_keys. If the key itself is in scaled_keys, it adds to its scaled value.
    Sets the result to the given agent's reward.
    :param Agent agent: the agent to set the reward.
    :param dict scaled_keys: the dictionary containing the weights (scalars) for each named key.
    :rtype: KeyedMatrix
    :return: a matrix performing the given linear combination to the given key.
    """
    return multi_set_matrix(rewardKey(agent.name), scaled_keys)


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


"""
    ACTION UTILITIES
"""


def set_action_legality(agent, action, legality=True, models=None):
    """
    Sets legality for an action for the given agent and model.
    :param Agent agent: the agent whose model(s) we want to set the action legality.
    :param ActionSet action: the action for which to set the legality.
    :param bool legality: whether to set this action legal (True) or illegal (False)
    :param list[str] models: the list of models for which to set the action legality. None will set to the agent itself.
    """
    # tests for "true" model
    if models is None or len(models) == 0:
        agent.setLegal(action, makeTree(legality))
        return

    model_key = modelKey(agent.name)

    # initial tree (end condition is: 'not legality')
    tree = not legality

    # recursively builds legality tree by comparing the model's key with the index of the model in the state/vector
    for model in models:
        tree = {'if': equalRow(model_key, agent.model2index(model)),
                True: legality,
                False: tree}
    agent.setLegal(action, makeTree(tree))


def set_illegal_action(agent, action, models=None):
    """
    Sets an illegal action for the given agent model only.
    :param Agent agent: the agent whose models we want to set the action legality.
    :param ActionSet action: the action for which to set the legality.
    :param list[str] models: the list of models for which to set the action legality. None will set to the agent itself.
    """
    set_action_legality(agent, action, False, models)


def set_legal_action(agent, action, models=None):
    """
    Sets a legal action for the given agent model only.
    :param Agent agent: the agent whose models we want to set the action legality.
    :param ActionSet action: the action for which to set the legality.
    :param list[str] models: the list of models for which to set the action legality. None will set to the agent itself.
    """
    set_action_legality(agent, action, True, models)


"""
    OTHER PSYCHSIM UTILITIES
"""


def get_true_model_name(agent):
    """
    Gets the name of the "True" model associated with the given agent.
    Node: this depends on the internals of PsychSim, so this function should probably be moved inside Agent.py.
    :param Agent agent: the agent whose true model we want to retrieve.
    :rtype: str
    :return: the name of true model of the given agent.
    """
    return '{}0'.format(agent.name)


def get_random_value(world, feature, rng=None):
    """
    Gets a random value for a given feature according to its domain.
    :param World world: the PsychSim world in which the feature is defined.
    :param str feature: the named feature.
    :param random.Random rng: the random state used to sample values.
    :return: a random value according to the feature's domain.
    """
    assert feature in world.variables, 'World does not contain feature \'{}\''.format(feature)

    var = world.variables[feature]
    domain = var['domain']
    rng = random.Random() if rng is None else rng

    if domain is float:
        return rng.uniform(var['lo'], var['hi'])
    if domain is int:
        return rng.randint(var['lo'], var['hi'])
    if domain is list or domain is set:
        return rng.choice(var['elements'])
    if domain is bool:
        return bool(rng.randint(0, 1))
    return None


"""
    NON-LINEAR DYNAMICS UTILITIES
"""


def get_univariate_samples(fnc, min_x, max_x, num_samples):
    """
    Creates samples for the given univariate function in some interval.
    :param callable fnc: the function to get samples from.
    :param float min_x: the minimal parameter value of the sample.
    :param float max_x: the maximal parameter value of the sample (inclusive).
    :param int num_samples: the number of samples to get from the function.
    :rtype: (np.ndarray, np.ndarray)
    :return: a tuple containing the parameters and the corresponding samples for the given function.
    """
    x_args = np.linspace(min_x, max_x, num_samples)
    return x_args, np.array([fnc(arg) for arg in x_args])


def get_bivariate_samples(fnc, min_x, max_x, min_y, max_y, num_x_samples, num_y_samples):
    """
    Creates samples for the given bivariate function in some interval.
    :param callable fnc: the function to get samples from.
    :param float min_x: the minimal x parameter value of the sample.
    :param float max_x: the maximal x parameter value of the sample (inclusive).
    :param float min_y: the minimal y parameter value of the sample.
    :param float max_y: the maximal y parameter value of the sample (inclusive).
    :param int num_x_samples: the number of samples to get from the function in the x-axis.
    :param int num_y_samples: the number of samples to get from the function in the y-axis.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    :return: a tuple containing the x parameters (num_x_samples, ), the y parameters (num_y_samples, ) and an array of
    shape (num_x_samples, num_y_samples) containing the corresponding sample values for the given function.
    """
    x_args = np.linspace(min_x, max_x, num_x_samples)
    y_args = np.linspace(min_y, max_y, num_y_samples)
    return x_args, y_args, np.array([[fnc(x, y) for y in y_args] for x in x_args])


def tree_from_univariate_samples(set_var, x_var, x_params, sample_values, idx_min=0, idx_max=-1):
    """
    Creates a PWL dynamics tree that sets the value of one feature according to the value of another as provided by a
    given set of samples. This a recursive function that creates a binary search tree to determine the "best match" for
    the value of the parameter feature.
    :param str set_var: the feature (named key) on which to store the approximation.
    :param str x_var: the feature (named key) providing the parameter value from which to calculate the approximation.
    :param np.ndarray x_params: an array of shape (num_samples, ) containing the values for the parameters.
    :param np.ndarray sample_values: an array of shape (num_samples, ) with the function values for each parameter value.
    :param int idx_min: the lower index of the current binary search.
    :param int idx_max: the upper index of the current binary search. -1 corresponds to num_samples - 1.
    :rtype: dict
    :return: a dictionary to be used with makeTree to define the dynamics of the approximation of the function that
    produced the given samples.
    """
    # checks indexes
    if idx_max == -1:
        idx_max = len(x_params) - 1

    # checks termination (leaf), sets to index's value
    if idx_min == idx_max:
        return setToConstantMatrix(set_var, sample_values[idx_max])

    # builds binary search tree
    idx = (idx_max + idx_min) // 2
    x = x_params[idx]
    return {'if': multi_compare_row({x_var: 1}, x),  # if var is greater than x
            True: tree_from_univariate_samples(
                set_var, x_var, x_params, sample_values, idx + 1, idx_max),  # search right
            False: tree_from_univariate_samples(
                set_var, x_var, x_params, sample_values, idx_min, idx)}  # search left


def tree_from_bivariate_samples(
        set_var, x_var, y_var, x_params, y_params, sample_values, idx_x_min=0, idx_x_max=-1, idx_y_min=0, idx_y_max=-1):
    """
    Creates a PWL dynamics tree that sets the value of one feature according to the value of two other as provided by a
    given set of samples. This a recursive function that creates two intertwined binary search trees to determine the
    "best match" for the value of the parameter feature pair.
    :param str set_var: the feature (named key) on which to store the approximation.
    :param str x_var: the feature (named key) providing the x-parameter value from which to calculate the approximation.
    :param str y_var: the feature (named key) providing the y-parameter value from which to calculate the approximation.
    :param np.ndarray x_params: an array of shape (num_x_samples, ) containing the values for the x parameters.
    :param np.ndarray y_params: an array of shape (num_y_samples, ) containing the values for the y parameters.
    :param np.ndarray sample_values: an array of shape (num_x_samples, num_y_samples) with the function values for each
    parameter pair.
    :param int idx_x_min: the lower x-index of the current binary search.
    :param int idx_x_max: the upper x-index of the current binary search. -1 corresponds to num_x_samples - 1.
    :param int idx_y_min: the lower y-index of the current binary search.
    :param int idx_y_max: the upper y-index of the current binary search. -1 corresponds to num_y_samples - 1.
    :rtype: dict
    :return: a dictionary to be used with makeTree to define the dynamics of the approximation of the function that
    produced the given samples.
    """
    # checks indexes
    if idx_x_max == -1:
        idx_x_max = len(x_params) - 1
    if idx_y_max == -1:
        idx_y_max = len(y_params) - 1

    # checks termination (leaf), sets to index's value
    if idx_x_min == idx_x_max and idx_y_min == idx_y_max:
        return setToConstantMatrix(set_var, sample_values[idx_x_max, idx_y_max])

    # tests for hyperplane in x_axis, performs binary search in y-axis
    if idx_x_min == idx_x_max:
        idx_y = (idx_y_max + idx_y_min) // 2
        y = y_params[idx_y]
        return {'if': multi_compare_row({y_var: -1}, -y),  # if y var is less than y
                True: tree_from_bivariate_samples(  # search left
                    set_var, x_var, y_var, x_params, y_params, sample_values, idx_x_min, idx_x_max, idx_y_min, idx_y),

                False: tree_from_bivariate_samples(  # search right
                    set_var, x_var, y_var, x_params, y_params, sample_values, idx_x_min, idx_x_max, idx_y + 1,
                    idx_y_max)}

    # otherwise performs binary search in x-axis
    idx_x = (idx_x_max + idx_x_min) // 2
    x = x_params[idx_x]
    return {'if': multi_compare_row({x_var: -1}, -x),  # if x var is less than x
            True: tree_from_bivariate_samples(  # search left
                set_var, x_var, y_var, x_params, y_params, sample_values, idx_x_min, idx_x, idx_y_min, idx_y_max),
            False: tree_from_bivariate_samples(  # search right
                set_var, x_var, y_var, x_params, y_params, sample_values, idx_x + 1, idx_x_max, idx_y_min, idx_y_max)}


"""
    DISCRETIZATION UTILITIES
"""


def discretize_feature_in_place(world, feature, num_bins):
    """
    Discretizes the given feature's value/distribution according to the number of intended groups in place, i.e.,
    by directly changing its value.
    :param World world: the PsychSim world in which the feature is defined.
    :param str feature: the named feature to be discretized.
    :param int num_bins: the number of discretization bins or buckets.
    :return:
    """
    variable = world.variables[feature]
    high = variable['hi']
    low = variable['lo']
    ran = float(high - low)
    dist = world.getFeature(feature)
    new_dist = Distribution()
    for val, prob in dist.items():
        val = int(round((float(val - low) / ran) * (num_bins - 1))) * (ran / (num_bins - 1)) + low
        new_dist[val] = prob
    world.setFeature(feature, new_dist)


def discretization_tree(world, feature, num_bins):
    """
    Creates a PWL dynamics tree that discretizes the given feature according to a number of bins.
    The discretized value corresponds to an approximation to the nearest discrete bin.
    :param World world: the world in which the feature is defined.
    :param str feature: the named feature to create the discretization tree.
    :param int num_bins: the number of bins to perform the discretization.
    :rtype: dict
    :return: a dictionary to be used with makeTree to define the dynamics of the discretization.
    """
    variable = world.variables[feature]
    high = variable['hi']
    low = variable['lo']
    samples = np.linspace(low, high, num_bins)
    return tree_from_univariate_samples(feature, feature, samples, samples)


"""
    EXPLANATION UTILITIES
"""


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
