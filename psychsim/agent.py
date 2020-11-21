from __future__ import print_function
import copy
import inspect
import logging
import math
import multiprocessing
import os
import random
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from xml.dom.minidom import Document,Node

from psychsim.action import Action,ActionSet
from psychsim.pwl import *
from psychsim.probability import Distribution

class Agent(object):
    """
    :ivar name: agent name
    :type name: str
    :ivar world: the environment that this agent inhabits
    :type world: L{World<psychsim.world.World>}
    :ivar actions: the set of possible actions that the agent can choose from
    :type actions: `Action<psychsim.action.Action>`
    :ivar legal: a set of conditions under which certain action choices are allowed (default is that all actions are allowed at all times)
    :type legal: L{ActionSet}S{->}L{KeyedPlane}
    :ivar omega: the set of observable state features
    :type ivar omega: {str}
    :ivar x: X coordinate to be used in UI
    :type x: int
    :ivar y: Y coordinate to be used in UI
    :type y: int
    :ivar color: color name to be used in UI
    :type color: str
    :ivar float belief_threshold: belief-update outcomes that have a likelihood belief this threshold are pruned (default is None, which means no pruning)
    """

    def __init__(self,name,world=None):
        self.world = world
        self.actions = set()
        self.legal = {}
        self.omega = True
#        self.O = True
        self.models = {}
        self.modelList = {}
        self.x = None
        self.y = None
        self.color = None
        if isinstance(name,Document):
            self.parse(name.documentElement)
        elif isinstance(name,Node):
            self.parse(name)
        else:
            self.name = name
        self.parallel = False
        self.epsilon = 1e-6

        self.belief_threshold = None

    """------------------"""
    """Policy methods"""
    """------------------"""
    def compilePi(self,model=None,horizon=None,debug=False):
        if model is None:
            model = self.models['%s0' % (self.name)]
        else:
            model = self.models[model]
        if 'V' not in model or horizon not in model['V']:
            self.compileV(model['name'],horizon,debug)
        if horizon is None:
            exit()
        policy = None
        for action,tree in model['V'][horizon].items():
            actionTree = tree.map(leafOp=lambda matrix: (matrix[rewardKey(self.name,True)],action))
            if policy is None:
                policy = actionTree
            else:
                policy = policy.max(actionTree)
            policy.prune(variables=self.world.variables)
        model['policy'][horizon] = policy.map(leafOp=lambda tup: tup[1])
        policy.prune(variables=self.world.variables)
        if debug:
            print(horizon)
            print(model['policy'][horizon])
        return model['policy'][horizon]
        
    def compileV(self,model=None,horizon=None,debug=False):
        self.world.dependency.getEvaluation()
        if model is None:
            model = self.models['%s0' % (self.name)]
        else:
            model = self.models[model]
        belief = self.getBelief(self.world.state,model['name'])
        if horizon is None:
            horizon = self.getAttribute('horizon',model['name'])
        R = self.getReward(model['name'])
        Rkey = rewardKey(self.name,True)
        actions = self.actions
        model['V'] = {}
        turns = sorted([(belief[k].first(),k) for k in belief.keys() if isTurnKey(k)])
        order = turns[:]
        for i in range(len(order)-1):
            assert order[i][0] < order[i+1][0],'Unable to project when actors act in parallel (%s and %s)' % \
                (state2agent(order[i][1]),state2agent(order[i+1][1]))
        while len(order) < horizon:
            order += turns
        order = [state2agent(entry[1]) for entry in order[:horizon]]
        for t in reversed(range(len(order))):
            subhorizon = len(order)-t
            other = self.world.agents[order[t]]
            if other.name == self.name:
                model['V'][subhorizon] = {}
                for action in actions:
                    if debug: 
                        print(action)
                    effects = self.world.deltaState(action,belief,belief.keys())
                    model['V'][subhorizon][action] = collapseDynamics(copy.deepcopy(R),effects)
#                    if debug: 
#                        print(model['V'][subhorizon][action])
                    if len(model['V'][subhorizon]) >= 3:
                        break
                if t > 0:
                    policy = self.compilePi(model['name'],subhorizon,debug)
                    exit()
            else:
                # Compile mental model of this agent's policy
                if debug:
                    print('Compiling horizon %d policy for %s' % (subhorizon,other.name))
                if modelKey(other.name) in belief:
                    mentalModel = self.world.getModel(other.name,belief)
                    assert len(mentalModel) == 1,'Currently unable to compile policies for uncertain mental models'
                    mentalModel = mentalModel.first()
                else:
                    models = [model for model in other.models.keys() if 'modelOf' not in model]
                    assert len(models) == 1,'Unable to compile policies without explicit mental model of %s' % (other.name)
                    mentalModel = models[0]
                # Distinguish my belief about this model from other agent's true model
                mentalModel = other.addModel('%s_modelOf_%s' % (self.name,mentalModel),
                                             parent=mentalModel,static=True)
                if len(other.actions) > 1:
                    # Possible decision
                    if 'horizon' in mentalModel:
                        subhorizon = min(mentalModel['horizon'],subhorizon)
                    pi = other.compilePi(mentalModel['name'],subhorizon,debug)
                    print(other.name,subhorizon)
                    raise RuntimeError
                else:
                    # Single action, no decision to be made
                    action = next(iter(other.actions))
                    effects = self.world.deltaState(action,belief,belief.keys())
                    mentalModel['policy'] = {0: collapseDynamics(copy.deepcopy(R),effects)}
                    self.world.setModel(other.name,mentalModel['name'],belief)
                if debug:
                    print(action)
                    print(mentalModel['policy'])
        return model['V'][horizon]
                            
    def decide(self,state=None,horizon=None,others=None,model=None,selection=None,actions=None,
               keySet=None,debug={}):
        """
        Generate an action choice for this agent in the given state

        :param state: the current state in which the agent is making its decision
        :type state: L{KeyedVector}
        :param horizon: the value function horizon (default is use horizon specified in model)
        :type horizon: int
        :param others: the optional action choices of other agents in the current time step
        :type others: strS{->}L{ActionSet}
        :param model: the mental model to use (default is model specified in state)
        :type model: str
        :param selection: how to translate value function into action selection
           - random: choose one of the maximum-value actions at random
           - uniform: return a uniform distribution over the maximum-value actions
           - distribution: return a distribution (a la quantal response or softmax) using rationality of the given model
           - consistent: make a deterministic choice among the maximum-value actions (default setting for a model)
           - ``None``: use the selection method specified by the given model (default)

        :type selection: str
        :param actions: possible action choices (default is all legal actions)
        :param keySet: subset of state features to project over (default is all state features)
        """
        if state is None:
            state = self.world.state
        if model is None:
            try:
                model = self.world.getModel(self.name,state)
            except KeyError:
                # Use real model as fallback?
                model = self.world.getModel(self.name)
        assert not model is True
#        print(self.name, model)
        if isinstance(model,Distribution):
            result = {}
            tree = None
            myAction = keys.stateKey(self.name,keys.ACTION)
            myModel = keys.modelKey(self.name)
            action_leaves = {}
            deterministic = True
            for submodel in model.domain():
                result[submodel] = self.decide(state,horizon,others,submodel,
                                               selection,actions,keySet)
                if isinstance(result[submodel]['action'],Distribution):
                    deterministic = False
                elif deterministic:
                    try:
                        action_leaves[result[submodel]['action']].add(submodel)
                    except KeyError:
                        action_leaves[result[submodel]['action']] = {submodel}
            if deterministic:
                if len(action_leaves) == 1:
                    # Only one possible action, no matter what model holds
                    tree = setToConstantMatrix(myAction, next(iter(action_leaves.keys())))
                else:
                    # Multiple branches necessary
                    tree = None
                    for action, submodels in action_leaves.items():
                        if tree is None:
                            tree = setToConstantMatrix(myAction, action)
                        else:
                            tree = {'if': equalRow(myModel, submodels),
                                True: setToConstantMatrix(myAction, action),
                                False: tree}
            else:
                for submodel in model.domain():
                    if isinstance(result[submodel]['action'],Distribution):
                        if len(result[submodel]['action']) > 1:
                            matrix = {'distribution': [(setToConstantMatrix(myAction,el),
                                                        result[submodel]['action'][el]) \
                                                       for el in result[submodel]['action'].domain()]}
                        else:
                            # Distribution with 100% certainty
                            matrix = setToConstantMatrix(myAction,result[submodel]['action'].first())
                    else:
                        matrix = setToConstantMatrix(myAction,result[submodel]['action'])
                    if tree is None:
                        # Assume it's this model (?)
                        tree = matrix
                    else:
                        plane = equalRow(myModel,submodel)
                        tree = {'if': plane, True: matrix,False: tree}
#            print(model)
#            print(tree)
            result['policy'] = makeTree(tree)
#            if len(model) == 1:
#                assert result['policy'].isLeaf()
            return result
        if selection is None:
            selection = self.getAttribute('selection',model)
        # What are my subjective beliefs for this decision?
        belief = self.getBelief(state,model)
        # Do I have a policy telling me what to do?
        policy = self.getAttribute('policy',model)
        if policy:
            assert len(belief) == 1,'Unable to apply PWL policies to uncertain beliefs'
            action = policy[iter(belief.domain()).next()]
            if action:
                if isinstance(action,Action):
                    action = ActionSet([action])
                return {'action': action}
        if horizon is None:
            horizon = self.getAttribute('horizon',model)
        else:
            horizon = min(horizon,self.getAttribute('horizon',model))
        if actions is None:
            # Consider all legal actions (legality determined by my belief, circumscribed by real world)
            actions = self.getLegalActions(belief)
        if len(actions) == 0:
            # Someone made a boo-boo because there is no legal action for this agent right now
            buf = StringIO()
            if len(self.getLegalActions(state)) == 0:
                print('%s [%s] has no legal actions in:' % (self.name,model),file=buf)
                self.world.printState(state,buf)
            else:
                print('%s has true legal actions:' % (self.name),\
                      ';'.join(map(str,sorted(self.getLegalActions(state)))),file=buf)
            if len(self.getLegalActions(belief)) == 0:
                print('%s has no legal actions when believing:' % (self.name),
                      file=buf)
                self.world.printState(belief,buf)
            else:
                print('%s believes it has legal actions:' % (self.name),\
                      ';'.join(map(str,sorted(self.getLegalActions(belief)))),file=buf)
            msg = buf.getvalue()
            buf.close()
            raise RuntimeError(msg)
        elif len(actions) == 1:
            # Only one possible action
            if selection == 'distribution':
                return {'action': Distribution({next(iter(actions)): 1.})}
            else:
                return {'action': next(iter(actions))}
        logging.debug('%s deciding...' % (self.name))
        # Keep track of value function
        Vfun = self.getAttribute('V',model)
        if Vfun:
            # Use stored value function
            V = {}
            for action in actions:
                b = copy.deepcopy(belief)
                b *= Vfun[action]
                V[action] = {'__EV__': b[rewardKey(self.name,True)].expectation()}
                logging.debug('Evaluated %s as %s (%d): %f' % (action, model, horizon, V[action]['__EV__']))
        elif self.parallel:
            with multiprocessing.Pool() as pool:
                results = [(action,pool.apply_async(self.value,
                                                    args=(belief,action,model,horizon,others,keySet)))
                           for action in actions]
                V = {action: result.get() for action,result in results}
        else:
            # Compute values in sequence
            V = {}
            for action in actions:
                V[action] = self.value(belief,action,model,horizon,others,keySet)
                logging.debug('Evaluated %s as %s (%d): %f' % (action, model, horizon,V[action]['__EV__']))
        best = None
        for action in actions:
            # Determine whether this action is the best
            if best is None:
                best = [action]
            elif V[action]['__EV__'] == V[best[0]]['__EV__']:
                best.append(action)
            elif V[action]['__EV__'] > V[best[0]]['__EV__']:
                best = [action]
        result = {'V*': V[best[0]]['__EV__'],'V': V}
        # Make an action selection based on the value function
        if selection == 'distribution':
            values = {}
            for key,entry in V.items():
                values[key] = entry['__EV__']
            result['action'] = Distribution(values,self.getAttribute('rationality',model))
        elif len(best) == 1:
            # If there is only one best action, all of the selection mechanisms devolve 
            # to the same unique choice
            result['action'] = best[0]
        elif selection == 'random':
            result['action'] = random.sample(best,1)[0]
        elif selection == 'uniform':
            result['action'] = {}
            prob = 1./float(len(best))
            for action in best:
                result['action'][action] = prob
            result['action'] = Distribution(result['action'])
        else:
            assert selection == 'consistent','Unknown action selection method: %s' % (selection)
            best.sort()
            result['action'] = best[0]
        logging.debug('Choosing %s' % (result['action']))
        return result

    def value(self,belief,action,model,horizon=None,others=None,keySet=None,updateBeliefs=True,
              debug={}):
        if horizon is None:
            horizon = self.getAttribute('horizon',model)
        if keySet is None:
            keySet = belief.keys()
        # Compute value across possible worlds
        logging.debug('Considering %s as %s' % (action,model))
        assert modelKey(self.name) in belief, '{} ({}) has no model of self when computing value of {}'.format(self.name, model, action)
        assert model in self.world.getFeature(modelKey(self.name), belief).domain(), '{}  not in {}'.format(model, str(self.world.getFeature(modelKey(self.name), belief).domain()))
        assert len(self.world.getFeature(modelKey(self.name), belief)) == 1
        current = copy.deepcopy(belief)
#        if model:
#            self.world.setFeature(modelKey(self.name),model,current)
        V_A = self.getAttribute('V',model)
        if V_A:
            current *= V_A[action]
            R = current[makeFuture(rewardKey(self.name))]
            V = {'__beliefs__': current,
                 '__S__': [current],
                 '__ER__': [R],
                 '__EV__': R.expectation()}
        else:
            V = {'__EV__': 0.,'__ER__': [],'__S__': []}
            if isinstance(keySet,dict):
                subkeys = keySet[action]
            else:
                subkeys = belief.keys()
            if others:
                start = dict(others)
            else:
                start = {}
            if action:
                start[self.name] = action
            for t in range(horizon):
                logging.debug('Time %d/%d' % (t+1,horizon))
                turn = self.world.next(current)
                actions = {}
                for name in turn:
                    if name in start:
                        actions[name] = start[name]
                        del start[name]
                outcome = self.world.step(actions,current,keySubset=subkeys,horizon=horizon-t,
                                          updateBeliefs=updateBeliefs,debug=debug)
                V['__ER__'].append(self.reward(current,model))
                V['__EV__'] += V['__ER__'][-1]
                V['__S__'].append(current)
            V['__beliefs__'] = current
        return V
        
    def oldvalue(self,vector,action=None,horizon=None,others=None,model=None,keys=None):
        """
        Computes the expected value of a state vector (and optional action choice) to this agent

        :param vector: the state vector (not distribution) representing the possible world under consideration
        :type vector: L{KeyedVector}
        :param action: prescribed action choice for the agent to evaluate; if ``None``, then use agent's own action choice (default is ``None``)
        :type action: L{ActionSet}
        :param horizon: the number of time steps to project into the future (default is agent's horizon)
        :type horizon: int
        :param others: optional table of actions being performed by other agents in this time step (default is no other actions)
        :type others: strS{->}L{ActionSet}
        :param model: the model of this agent to use (default is ``True``)
        :param keys: subset of state features to project over in computing future value (default is all state features)
        """
        if model is None:
            model = self.world.getModel(self.name,vector)
        # Determine horizon
        if horizon is None:
            horizon = self.getAttribute('horizon',model)
        # Determine discount factor
        discount = self.getAttribute('discount',model)
        # Compute immediate reward
        R = self.reward(vector,model)
        result = {'R': R,
                  'agent': self.name,
                  'state': vector,
                  'horizon': horizon,
                  'projection': []}
        # Check for pre-computed value function
        V = self.getAttribute('V',model).get(self.name,vector,action,horizon,
                                             self.getAttribute('ignore',model))
        if V is not None:
            result['V'] = V
        else:
            result['V'] = R
            if horizon > 0 and not self.world.terminated(vector):
                # Perform action(s)
                if others is None:
                    turn = {}
                else:
                    turn = copy.copy(others)
                if not action is None:
                    turn[self.name] = action
                outcome = self.world.stepFromState(vector,turn,horizon,keySubset=keys)
                if not 'new' in outcome:
                    # No consistent outcome
                    pass
                elif isinstance(outcome['new'],Distribution):
                    # Uncertain outcomes
                    future = Distribution()
                    for newVector in outcome['new'].domain():
                        entry = copy.copy(outcome)
                        entry['probability'] = outcome['new'][newVector]
                        Vrest = self.value(newVector,None,horizon-1,None,model,keys)
                        entry.update(Vrest)
                        try:
                            future[entry['V']] += entry['probability']
                        except KeyError:
                            future[entry['V']] = entry['probability']
                        result['projection'].append(entry)
                    # The following is typically "expectation", but might be "max" or "min", too
                    op = self.getAttribute('projector',model)
                    if discount < -self.epsilon:
                        # Only final value matters
                        result['V'] = apply(op,(future,))
                    else:
                        # Accumulate value
                        result['V'] += discount*apply(op,(future,))
                else:
                    # Deterministic outcome
                    outcome['probability'] = 1.
                    Vrest = self.value(outcome['new'],None,horizon-1,None,model,keys)
                    outcome.update(Vrest)
                    if discount < -self.epsilon:
                        # Only final value matters
                        result['V'] = Vrest['V']
                    else:
                        # Accumulate value
                        result['V'] += discount*Vrest['V']
                    result['projection'].append(outcome)
            # Do some caching
            self.getAttribute('V',model).set(self.name,vector,action,horizon,result['V'])
        return result

    def valueIteration(self,horizon=None,ignore=None,model=True,epsilon=1e-6,debug=0,maxIterations=None):
        """
        Compute a value function for the given model
        """
        if horizon is None:
            horizon = self.getAttribute('horizon',model)
        if ignore is None:
            ignore = self.getAttribute('ignore',model)
        # Find transition matrix
        transition = self.world.reachable(horizon=horizon,ignore=ignore,debug=(debug > 1))
        if debug:
            print('|S|=%d' % (len(transition)))
        # Initialize value function
        V = self.getAttribute('V',model)
        newChanged = set()
        for start in transition.keys():
            for agent in self.world.agents.values():
                if self.world.terminated(start):
                    if agent.name == self.name:
                        value = agent.reward(start,model)
                    else:
                        value = agent.reward(start)
                    V.set(agent.name,start,None,0,value)
                    if abs(value) > epsilon:
                        newChanged.add(start)
                else:
                    V.set(agent.name,start,None,0,0.)
        # Loop until no change in value function
        iterations = 0
        while len(newChanged) > 0 and (maxIterations is None or iterations < maxIterations):
            iterations += 1
            if debug > 0:
                print('Iteration %d' % (iterations))
            oldChanged = newChanged.copy()
            newChanged.clear()
            recomputed = set()
            newV = ValueFunction()
            # Consider all possible nodes whose value has changed on the previous iteration
            for node in oldChanged:
                if debug > 1:
                    print
                    self.world.printVector(node)
                for start in transition[node]['__predecessors__'] - recomputed:
                    recomputed.add(start)
                    # This is a state whose value might have changed
                    actor = None
                    for action,distribution in transition[start].items():
                        if action == '__predecessors__':
                            continue
                        if debug > 2:
                            print('\t\t%s' % (action))
                        # Make sure only one actor is acting at a time
                        if actor is None:
                            actor = action['subject']
                        else:
                            assert action['subject'] == actor,'Unable to do value iteration with concurrent actors'
                        # Consider all possible results of this action
                        for agent in self.world.agents.values():
                            # Accumulate expected rewards from possible transitions
                            ER = 0.
                            for end in distribution.domain():
                                # Determine expected value of future
                                future = V.get(agent.name,end,None,0)
                                if future is None:
                                    Vrest = 0.
                                else:
                                    Vrest = distribution[end]*future
                                # Determine discount function 
                                # (should use belief about other agent, but doesn't yet)
                                if agent.name == self.name:
                                    discount = agent.getAttribute('discount',model)
                                else:
                                    discount = agent.getAttribute('discount',True)
                                if discount < -epsilon:
                                    # Future reward is all that matters
                                    ER += distribution[end]*Vrest
                                else:
                                    # Current reward + Discounted future reward
                                    if agent.name == self.name:
                                        R = agent.reward(start,model)
                                    else:
                                        R = agent.reward(start)
                                    ER += distribution[end]*(R+discount*Vrest)
                            newV.set(agent.name,start,action,0,ER)
                            if debug > 2:
                                print('\t\t\tV_%s = %5.3f' % (agent.name,ER))
                    # Value of state is the value of the chosen action in this state
                    choice = self.predict(start,actor,newV,0)
                    if debug > 2:
                        print('\tPrediction\n%s' % (choice))
                    delta = 0.
                    for name in self.world.agents.keys():
                        for action in choice.domain():
                            newV.add(name,start,None,0,choice[action]*newV.get(name,start,action,0))
                        old = V.get(name,start,None,0)
                        if old is None:
                            delta += abs(newV.get(name,start,None,0))
                        else:
                            delta += abs(newV.get(name,start,None,0) - old)
                        if debug > 1:
                            print('\tV_%s = %5.3f' % (name,newV.get(name,start,None,0)))
                    if delta > epsilon:
                        newChanged.add(start)
            V = newV
            self.setAttribute('V',V,model)
        if debug > 0:
            print('Completed after %d iterations' % (iterations))
        return self.getAttribute('V',model)

    def setPolicy(self,policy,model=None):
        self.setAttribute('policy',policy.desymbolize(self.world.symbols),model)

    def setHorizon(self,horizon,model=None):
        """
        :type horizon: int
        :param model: the model to set the horizon for, where ``None`` means set it for all (default is ``None``)
        """
        self.setAttribute('horizon',horizon,model)

    def setParameter(self,name,value,model=None):
        raise DeprecationWarning('Use setAttribute instead')

    def setAttribute(self,name,value,model=None):
        """
        Set a parameter value for the given model(s)
        :param name: the feature of the model to set
        :type name: str
        :param value: the new value for the parameter
        :param model: the model to set the horizon for, where ``None`` means set it for all (default is ``None``)
        """
        if model is None:
            for model in self.models.values():
                self.setAttribute(name,value,model['name'])
        else:
            self.models[model][name] = value

    def findAttribute(self,name,model):
        """
        
    :returns: the name of the nearest ancestor model (include the given model itself) that specifies a value for the named feature
        """
        if name in self.models[model]:
            return model
        elif self.models[model]['parent'] is None:
            return None
        else:
            return self.findAttribute(name,self.models[model]['parent'])

    def getAttribute(self,name,model):
        """
        
    :returns: the value for the specified parameter of the specified mental model
        """
        ancestor = self.findAttribute(name,model)
        if ancestor is None:
            return None
        else:
            return self.models[ancestor][name]

    """------------------"""
    """Action methods"""
    """------------------"""

    def addAction(self,action,condition=None,description=None,codePtr=True):
        """
        :param condition: optional legality condition
        :type condition: L{KeyedPlane}
        :returns: the action added
        :rtype: L{ActionSet}
        """
        actions = []
        if isinstance(action,set) or isinstance(action,frozenset) or isinstance(action,list):
            for atom in action:
                if isinstance(atom,Action):
                    actions.append(Action(atom))
                else:
                    actions.append(atom)
        elif isinstance(action,Action):
            actions.append(action)
        else:
            assert isinstance(action,dict),'Argument to addAction must be at least a dictionary'
            actions.append(Action(action,description))
        for atom in actions:
            if not 'subject' in  atom:
                # Make me the subject of these actions
                atom['subject'] = self.name
        new = ActionSet(actions)
        assert new not in self.actions,'Action %s already defined' % (new)
        self.actions.add(new)
        if condition:
            self.legal[new] = condition.desymbolize(self.world.symbols)
        if codePtr:
            if codePtr is True:
                for frame in inspect.getouterframes(inspect.currentframe()):
                    try:
                        fname = frame.filename
                    except AttributeError:
                        fname = frame[1]
                    if fname != __file__:
                        break
            else:
                frame = codePtr
            mod = os.path.relpath(frame.filename,
                                  os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
            try:
                self.world.extras[new] = '%s:%d' % (mod,frame.lineno)
            except AttributeError:
                self.world.extras[new] = '%s:%d' % (mod,frame[2])
        # Add to state vector
        key = actionKey(self.name)
        if key in self.world.variables:
            self.world.symbols[new] = len(self.world.symbols)
            self.world.symbolList.append(new)
            self.world.variables[key]['elements'].add(new)
        else:
            self.world.defineVariable(key,ActionSet,description='Action performed by %s' % (self.name))
            self.world.setFeature(key,new)
        self.world.dynamics[new] = {}
        return new

    def getActions(self,vector=None,actions=None):
        raise DeprecationWarning('This method has been renamed "getLegalActions"')

    def getLegalActions(self,vector=None,actions=None):
        """
        :param vector: the world in which to test legality
        :param actions: the set of actions to test legality of (default is all available actions)
        
    :returns: the set of possible actions to choose from in the given state vector
        :rtype: {L{ActionSet}}
        """
        if vector is None:
            vector = self.world.state
        if actions is None:
            actions = self.actions
        if len(self.legal) == 0:
            # No restrictions on legal actions, so take a shortcut
            return actions
        # Otherwise, filter out illegal actions
        result = set()
        for action in actions:
            try:
                tree = self.legal[action]
            except KeyError:
                # No condition on this action's legality => legal
                result.add(action)
                continue
            # Must satisfy all conditions
            if tree[vector]:
                result.add(action)
        return result

    def setLegal(self,action,tree):
        """
        Sets the legality decision tree for a given action
        :param action: the action whose legality we are setting
        :param tree: the decision tree for the legality of the action
        :type tree: L{KeyedTree}
        """
        self.legal[action] = tree.desymbolize(self.world.symbols)

    def hasAction(self,atom):
        """
        :type atom: L{Action}
        
    :returns: ``True`` iff this agent has the given action (possibly in combination with other actions)
        :rtype: bool
        """
        for action in self.actions:
            for candidate in action:
                if atom.root() == candidate.root():
                    return True
        else:
            return False

    """------------------"""
    """State methods"""
    """------------------"""

    def setState(self,feature,value,state=None):
        return self.world.setState(self.name,feature,value,state)

    def getState(self,feature,state=None,unique=False):
        return self.world.getState(self.name,feature,state,unique)

    """------------------"""
    """Reward methods"""
    """------------------"""

    def setReward(self,tree,weight=0.,model=None):
        """
        Adds/updates a goal weight within the reward function for the specified model.
        """
        if model is None:
            for model in self.world.getModel(self.name,self.world.state).domain():
                self.setReward(tree,weight,model)
        else:
            if self.models[model].get('R',None) is None:
                self.models[model]['R'] = {}
            if not isinstance(tree,str):
                tree = tree.desymbolize(self.world.symbols)
            self.models[model]['R'][tree] = weight
            key = rewardKey(self.name)
            if not key in self.world.variables:
                self.world.defineVariable(key,float,
                                          description='Reward for %s in this state' % (self.name))
                self.world.setFeature(key,0.)

    def getReward(self,model=None):
        if model is None:
            model = self.world.getModel(self.name,self.world.state)
            if isinstance(model,Distribution):
                return {m: self.getReward(m) for m in model.domain()}
            else:
                return {model: self.getReward(model)}
        R = self.getAttribute('R',model)
        if R is None:
            # No reward components
#            R = KeyedTree(setToConstantMatrix(rewardKey(self.name),0.))
#            self.setAttribute('R',R,model)
            return R
        elif isinstance(R,dict):
            Rsum = None
            for tree,weight in R.items():
                if isinstance(tree,str):
                    agent = self.world.agents[tree]
                    dist = self.world.getModel(agent.name,self.getBelief(model=model))
                    if len(dist) == 1:
                        otherModel = dist.first()
                        tree = agent.getReward(otherModel)
                    else:
                        raise NotImplementedError('Simple fix needed to support agents having rewards tied to other agents about whom they have uncertain beliefs')
                if Rsum is None:
                    Rsum = weight*tree
                else:
                    Rsum += weight*tree
            if Rsum is None:
                Rsum = KeyedTree(setToConstantMatrix(rewardKey(self.name),0.))
            self.setAttribute('R',Rsum,model)
            return Rsum
        else:
            return R
        
    def reward(self,vector=None,model=None,recurse=True):
        """
        :param recurse: ``True`` iff it is OK to recurse into another agent's reward (default is ``True``)
        :type recurse: bool
        
    :returns: the reward I derive in the given state (under the given model, default being the ``True`` model)
        :rtype: float
        """
        total = 0.
        if vector is None:
            total = self.reward(self.world.state,model,recurse)
        elif isinstance(vector,VectorDistribution):
            for element in vector.domain():
                total += vector[element]*self.reward(element,model,recurse)
        elif isinstance(vector,VectorDistributionSet):
            if model is None:
                modelK = modelKey(self.name)
                models = self.world.float2value(modelK,vector.domain(modelK))
                tree = None
                for submodel in models:
                    R = self.getReward(submodel)
                    if tree is None:
                        tree = R
                    else:
                        tree = {'if': equalRow(modelK,submodel),
                                 True: R,False: tree}
                tree = makeTree(tree).desymbolize(self.world.symbols)
            else:
                tree = self.getReward(model)
            if tree is None:
                raise ValueError('Agent "{} has no reward function defined (model "{}")'.format(self.name, model))
            vector *= tree
            if not rewardKey(self.name) in vector:
                vector.join(rewardKey(self.name),0.)
            vector.rollback()
            total = vector[rewardKey(self.name)].expectation()
        else:
            tree = self.getReward(model)
            vector *= tree
            vector.rollback()
            total = float(vector[rewardKey(self.name)])
        return total

    def printReward(self,model=True,buf=None,prefix=''):
        first = True
        R = self.getAttribute('R',model)
        if isinstance(R,dict):
            for tree,weight in R.items():
                if first:
                    msg = '%s\tR\t\t%3.1f %s' % (prefix,weight,str(tree))
                    print(msg.replace('\n','\n%s\t\t\t' % (prefix)),file=buf)
                    first = False
                else:
                    msg = '%s\t\t\t%3.1f %s' % (prefix,weight,str(tree))
                    print(msg.replace('\n','\n%s\t\t\t' % (prefix)),file=buf)
        else:
            msg = '%s\tR\t\t%s' % (prefix,str(R))
            print(msg.replace('\n','\n%s\t\t\t' % (prefix)),file=buf)


    """------------------"""
    """Mental model methods"""
    """------------------"""

    def ignore(self,agents,model=None):
        try:
            beliefs = self.models[model]['beliefs']
        except KeyError:
            beliefs = True
        if beliefs is True:
            beliefs = self.resetBelief(model)
        if isinstance(agents,str):
            for key in list(beliefs.keys()):
                if state2agent(key) == agents:
                    del beliefs[key]
#            del beliefs[keys.turnKey(agents)]
#            del beliefs[keys.modelKey(agents)]
        else:
            beliefs.deleteKeys([keys.turnKey(a) for a in agents]+
                               [keys.modelKey(a) for a in agents])

    def addModel(self,name,**kwargs):
        """
        Adds a new possible model for this agent (to be used as either true model or else as mental model another agent has of it). Possible arguments are:
         - R: the reward table for the agent under this model (default is ``True``), L{KeyedTree}S{->}float
         - beliefs: the beliefs the agent has under this model (default is ``True``), L{MatrixDistribution}
         - horizon: the horizon of the value function under this model (default is ``True``),int
         - rationality: the rationality parameter used in a quantal response function when modeling others (default is 10),float
         - discount: discount factor used in lookahead
         - selection: selection mechanism used in L{decide}
         - parent: another model that this model inherits from (default is ``True``)

        :param name: the label for this model
        :type name: sotr
        
        :returns: the model created
        :rtype: dict
        """
        if name is None:
            raise NameError('"None" is an illegal model name')
        if name in self.models:
            return self.models[name]
#        if name in self.world.symbols:
#            raise NameError('Model %s conflicts with existing symbol' % (name))
        model = {'name': name,'index': 0,'parent': None,'SE': {}, 'transition': {},
#                 'V': ValueFunction(),
                 'policy': {},'ignore': []}
        model.update(kwargs)
        model['index'] = len(self.world.symbolList)
        self.models[name] = model
        self.modelList[model['index']] = name
        self.world.symbols[name] = model['index']
        self.world.symbolList.append(name)
        if not name in self.world.variables[modelKey(self.name)]['elements']:
            self.world.variables[modelKey(self.name)]['elements'].append(name)
        return model

    def get_true_model(self,unique=True):
        """
        :return: the name of the "true" model of this agent, i.e., the model by which the real agent is governed in the real world
        :rtype: str
        :param unique: If True, assume there is a unique true model (default is True)
        :type unique: bool
        """
        return self.world.getModel(self.name,unique=unique)

    def deleteModel(self,name):
        """
        Deletes the named model from the space

        .. warning:: does not check whether there are remaining references to this model
        """
        del self.modelList[self.models[name]['index']]
        del self.models[name]

    def predict(self,vector,name,V,horizon=0):
        """
        Generate a distribution over possible actions based on a table of values for those actions
        :param V: either a L{ValueFunction} instance, or a dictionary of float values indexed by actions
        :param vector: the current state vector
        :param name: the name of the agent whose behavior is to be predicted
        """
        if isinstance(V,ValueFunction):
            V = V.actionTable(name,vector,horizon)
        choices = Distribution()
        if name == self.name:
            # I predict myself to maximize
            best = None
            for action,value in V.items():
                if best is None or value > best:
                    best = value
            best = filter(lambda a: V[a] == best,V.keys())
            for action in best:
                choices[action] = 1./float(len(best))
        else:
            rationality = self.world.agents[name].getAttribute('rationality',
                                                               self.world.getModel(name,vector))
            choices = Distribution(V,rationality)
        return choices

    def expectation(self,other,model=None,state=None):
        """
        :return: what I expect this other agent to do
        """
        if state is None:
            state = self.world.state
        if model is None:
            models = self.world.getModel(self.name).domain()
        elif isinstance(model,str):
            models = [model]
        result = {}
        for myModel in models:
            result[myModel] = {}
            beliefs = self.models[myModel]['beliefs']
            dist = self.world.getFeature(modelKey(other),beliefs)
            for yrModel in dist.domain():
                result[myModel][yrModel] = {'probability': dist[yrModel]}
                result[myModel][yrModel]['decision'] = self.world.agents[other].decide(state,model=yrModel)
        return result

    def model2index(self,model):
        """
        Convert a model name to a numeric representation
        :param model: the model name
        :type model: str
        :rtype: int
        """
        return self.models[model]['index']

    def index2model(self,index,throwException=False):
        """
        Convert a numeric representation of a model to a name
        :param index: the numeric representation of the model
        :type index: int
        :rtype: str
        """
        if isinstance(index,float):
            index = int(index+0.5)
        try:
            return self.modelList[index]
        except KeyError:
            # Unknown model index (hopefully, because of explaining post-GC)
            if throwException:
                raise IndexError('Unknown model index %s of %s' % (index,self.name))
            else:
                return None

    def belief2model(self,parent,belief):
        # Find "root" model (i.e., one that has more than just beliefs)
        if not isinstance(parent,dict):
            parent = self.models[parent]
        while not 'R' in parent and not parent['parent'] is None:
            # Find the model from which we inherit reward
            parent = self.models[parent['parent']]
        # Check whether this is even a new belief (the following loop does badly otherwise)
        if 'beliefs' in parent and parent['beliefs'] == belief:
            return parent
        # Find model sharing same parent that has same beliefs
#        for model in filter(lambda m: m['parent'] == parent['name'],self.models.values()):
#            if 'beliefs' in model and not model['beliefs'] is True:
#                if model['beliefs'] == belief:
#                    return model
#        else:
        # Create a new model
        name = '%s%d' % (parent['name'], hash(str(belief)))
        if name in self.models:
            for key in belief.keys():
                if belief[key] != self.models[name]['beliefs'][key]:
                    assert key == modelKey(self.name), 'Hash collision for {}!'.format(self.name)
            else:
                return self.models[name]
        return self.addModel(name, beliefs=belief, parent=parent['name'])

    def printModel(self,model=None,buf=None,index=None,prefix='',reward=False,previous=None):
        if isinstance(index,int) or isinstance(index,float):
            model = self.index2model(index)
        if model is None:
            return
        if not isinstance(model,dict):
            model = self.models[model]
        if previous is None or model['name'] not in previous:
            # Have not printed out this model before
            if isinstance(previous,set):
                previous.add(model['name'])
            if ('R' in model and model['R'] is not None) or 'beliefs' in model:
                print('%s%s=%s' % (prefix,self.name,model['name']),file=buf)
                if reward and 'R' in model and model['R'] is not None:
                    self.printReward(model['name'],buf,'%s\t\t' % (prefix))
                if 'beliefs' in model and not model['beliefs'] is True:
                    print('%s\t\t\tB' % (prefix),file=buf)
                    self.world.printState(model['beliefs'],buf,prefix+'\t\t\t',beliefs=True,models=previous)
        
    """---------------------"""
    """Belief update methods"""
    """---------------------"""

    def resetBelief(self,state=None,model=None,include=None,ignore=None,stateType=VectorDistributionSet):
        """
        Handles all combinations of state type and specified belief type
        """
        assert ignore is None or include is None,'Use either ignore or include sets, but not both'
        if state is None:
            state = self.world.state
        if model is None:
            assert len(self.models) == 1,'Model is unspecified and ambiguous'
            model = next(iter(self.models.keys()))
        if isinstance(state,VectorDistributionSet):
            if issubclass(stateType,VectorDistributionSet):
                beliefs = state.copySubset(ignore,include)
            elif issubclass(stateType,KeyedVector):
                vector = state.vector()
                beliefs = stateType({key: vector[key] for key in include if key not in ignore})
                assert CONSTANT in beliefs
            else:
                assert issubclass(stateType,VectorDistribution),'Unknown type %s specified for %s beliefs' % (stateType.__name__,self.name)
                beliefs = stateType()
                for vector in state:
                    beliefs.addProb(KeyedVector({key: vector[key] for key in include if key not in ignore}),prob)
        elif isinstance(state,KeyedVector):
            if ignore is None:
                ignore = set()
            if include is None:
                include = state.keys()
            if issubclass(stateType,KeyedVector):
                beliefs = stateType({key: state[key] for key in include if key not in ignore})
            elif issubclass(stateType,VectorDistribution):
                beliefs = stateType({KeyedVector({key: state[key] for key in include if key not in ignore}): 1})
            else:
                assert issubclass(stateType,VectorDistributionSet),'Unknown type %s specified for %s beliefs' % (stateType.__name__,self.name)
                beliefs = stateType()
                for key in include:
                    if key not in ignore:
                        beliefs.join(key,state[key])
        else:
            if ignore is None:
                ignore = set()
            if include is None:
                include = state.keys()
            assert issubclass(state.__class__,VectorDistribution),'Unable to extract beliefs from state of type %s ' % (stateType.__name__)
            if issubclass(stateType,VectorDistributionSet):
                dist = state.__class__()
                for vector in state.domain():
                    dist.addProb(vector.__class__({key: vector[key] for key in include if key not in ignore}),state[vector])
                beliefs = stateType(copy.deepcopy(dist))
            elif issubclass(stateType,KeyedVector):
                beliefs = stateType()
                for key in include:
                    if key not in ignore:
                        value = state.marginal(key)
                        assert len(value) == 1,'Unable to identify unique value for %s for %s beliefs' % (key,self.name)
                        beliefs[key] = value.first()
            else:
                assert issubclass(stateType,VectorDistribution),'Unknown type %s specified for %s beliefs' % (stateType.__name__,self.name)
                beliefs = stateType()
                for vector in state.domain():
                    beliefs.addProb(vector.__class__({key: vector[key] for key in include if key not in ignore}),state[vector])
        if (include is None or modelKey(self.name) in include) and (ignore is None or modelKey(self.name) not in ignore):
            self.world.setFeature(modelKey(self.name), model, beliefs)
        self.models[model]['beliefs'] = beliefs
        return beliefs
        
    def setRecursiveLevel(self,level,model=None):
        raise DeprecationWarning('Setting level of recursion is no longer supported. Explicitly specify what the nested mental models are.')

    def setBelief(self,key,distribution,model=None,state=None):
        if state is None:
            state = self.world.state
        if model is None:
            dist = self.world.getModel(self.name,state)
            for model in dist.domain():
                self.setBelief(key,distribution,model,state)
        try:
            beliefs = self.models[model]['beliefs']
        except KeyError:
            beliefs = True
        if beliefs is True:
            beliefs = self.resetBelief(state,model)
        self.world.setFeature(key,distribution,beliefs)

    def getBelief(self,vector=None,model=None):
        """
        :param model: the model of the agent to use, default is to use model specified in the state vector
        :returns: the agent's belief in the given world
        """
        if vector is None:
            vector = self.world.state
        if model is None:
            model = self.world.getModel(self.name,vector)
        if isinstance(model,Distribution):
            return {element: self.getBelief(vector,element) \
                    for element in model.domain()}
        else:
            beliefs = self.getAttribute('beliefs',model)
            if beliefs.__class__ is dict:
                logging.warning('%s has extraneous layer of nesting in beliefs' % (self.name))
                beliefs = beliefs[model]
            if beliefs is True:
                world = copy.deepcopy(vector)
            else:
                world = beliefs # copy.deepcopy(beliefs)
            return world

    def updateBeliefs(self,state=None,actions=set(),horizon=None):
        if state is None:
            state = self.world.state
        if isinstance(state,KeyedVector):
            model = self.stateEstimator(state,actions,horizon)
            vector[modelKey(self.name,True)] = self.world.value2float(modelKey(self.name),model)
        else:
            self.updateBeliefsOLD(state,actions,horizon)

    def stateEstimator(self,state,actions,horizon=None):
        if not isinstance(state,KeyedVector):
            raise TypeError('Operates on only KeyedVector instances')
        oldModel = self.world.getFeature(modelKey(self.name),state)
        if self.getAttribute('static',oldModel) is True:
            # My beliefs (and my current mental model) never change
            newModel = state[modelKey(self.name)]
        else:
            SE = self.models[oldModel]['SE']
            myAction = ActionSet({action for action in actions if action['subject'] == self.name})
            omega = ','.join(['%s' % (state[o]) for o in self.omega])
            if omega not in SE:
                SE[omega] = {}
            if myAction not in SE[omega]:
                SE[omega][myAction] = {}
            try:
                newModel = SE[omega][myAction][horizon]
                if newModel is None:
                    # We're still processing
                    newModel = self.models[oldModel]['index']
                else:
                    # We've finished processing this belief update
                    newModel = self.models[newModel]['index']
            except KeyError:
                pass
            if myAction in self.models[oldModel]['SE'] and label in self.models[oldModel]['SE'][myAction]:
                newModel = self.models[oldModel]['SE'][myAction][label]
                if newModel is None:
                    pass
            else:
                # Work to be done. First, mark that we've started processing this transition
                if myAction not in self.models[oldModel]['SE']:
                    self.models[oldModel]['SE'] = {myAction: {}}
                self.models[oldModel]['SE'][myAction][label] = None
                # Get old belief state.
                beliefs = copy.deepcopy(original)
                # Project direct effect of the actions, including possible observations
                outcome = self.world.step({self.name: myAction} if myAction else None,beliefs,
                    keySubset=beliefs.keys(),horizon=horizon,updateBeliefs=False)
                # Condition on actual observations
                for omega in self.omega:
                    value = vector[omega]
                    if not omega in beliefs:
                        continue
                    for b in beliefs.distributions[beliefs.keyMap[omega]].domain():
                        if b[omega] == value:
                            break
                    else:
                        if omega == oldModelKey:
                            continue
                        else:
                            logging.warning('%s (model %s) has impossible observation %s=%s when doing %s' % \
                                          (self.name,oldModel,omega,self.world.float2value(omega,vector[omega]),myAction))
                            SE[oldModel][label] = None
                            break
                    beliefs[omega] = vector[omega]
                else:
                    # Create model with these new beliefs
                    # TODO: Look for matching model?
                    for dist in beliefs.distributions.values():
                        if len(dist) > 1:
                            deletion = False
                            for vec in dist.domain():
                                if dist[vec] < self.epsilon:
                                    del dist[vec]
                                    deletion = True
                            if deletion:
                                dist.normalize()
                    newModel = self.belief2model(oldModel,beliefs)
                    SE[oldModel][label] = newModel['index']
                    if oldModelKey in self.omega:
                        # Observe this new model
                        beliefs.join(oldModelKey,newModel['index'])
                    self.models[oldModel]['SE'][myAction][label] = newModel['name']
            if SE[oldModel][label] is not None:
                # Insert new model into true state
                if isinstance(SE[oldModel][label],int) or isinstance(SE[oldModel][label],float):
                    vector[newModelKey] = SE[oldModel][label]
                else:
                    raise RuntimeError('Unable to process stochastic belief updates:%s' \
                        % (SE[oldModel][olabel]))
                newDist.addProb(vector,prob)
        newDist.normalize()
#        assert len(newDist) > 0
#        for vector in newDist.domain():
#            assert newModelKey in vector
#            newModel = self.world.float2value(modelKey(self.name),vector[newModelKey])
#            newBelief = self.getBelief(model=newModel)
        return model

    def updateBeliefsOLD(self,trueState=None,actions={},horizon=None):
        """
        .. warning:: Even if this agent starts with ``True`` beliefs, its beliefs can deviate after actions with stochastic effects (i.e., the world transitions to a specific state with some probability, but the agent only knows a posterior distribution over that resulting state). If you want the agent's beliefs to stay correct, then set the ``static`` attribute on the model to ``True``.

        """
        if trueState is None:
            trueState = self.world.state
        oldModelKey = modelKey(self.name)
        newModelKey = makeFuture(oldModelKey)
        # Find distribution over current belief models
        substate = trueState.keyMap[oldModelKey]
        trueState.keyMap[newModelKey] = substate
        oldDist = trueState.distributions[substate]
        newDist = oldDist.__class__()
        trueState.distributions[substate] = newDist
        for vector,prob in [(vector,oldDist[vector]) for vector in oldDist.domain()]:
            oldModel = self.world.float2value(oldModelKey,vector[oldModelKey])
            logging.debug('{} updating beliefs {} under model {} (horizon={})'.format(self.name, str(vector), oldModel, horizon))
            if self.getAttribute('static',oldModel) is True:
                # My beliefs (and my current mental model) never change
                newModel = oldModel
            elif self.omega is True:
                # My beliefs change, but they are accurate
                old_beliefs = self.models[oldModel]['beliefs']
                new_beliefs = trueState.copySubset(include=old_beliefs.keys()-vector.keys())
                for key in vector.keys():
                    if key == oldModelKey:
                        newModel = self.belief2model(oldModel, new_beliefs)['name']
                        self.world.setFeature(oldModelKey, newModel, new_beliefs)
                    elif key != CONSTANT:
                        self.world.setFeature(key, vector[key], new_beliefs)
            else:
                SE = self.models[oldModel]['SE']
                logging.debug('SE({}): {}'.format(oldModel, SE))
                P = {} # self.models[oldModel]['transition']
                # Identify label for overall observation
                omega = tuple([vector[o] for o in self.omega])
                if omega not in SE:
                    SE[omega] = {}
                if self.name in actions:
                    myAction = self.world.float2value(actionKey(self.name),vector[actionKey(self.name)])
                else:
                    myAction = None
                if myAction not in P:
                    P[myAction] = {}
                if myAction not in SE[omega]:
                    SE[omega][myAction] = {}
                logging.debug('{} {} {} {} {}'.format(self.name, oldModel, omega, myAction, horizon))
                for temp_action, table in P.items():
                    logging.debug('{}: {}'.format(temp_action, str(table)))
                if horizon in SE[omega][myAction]:
                    newModel = SE[omega][myAction][horizon]
                    if newModel is None:
                        # Processing this somewhere above me in the recursion
                        logging.debug('Avoid redundancy...')
                        newModel = oldModel
                else:
                    # Work to be done. First, mark that we've started processing this transition
                    SE[omega][myAction][horizon] = None
                    try:
                        original = None
                        beliefs = copy.deepcopy(P[myAction][horizon])
                        logging.debug('Retrieved step result')
                    except KeyError:
                        original = self.getBelief(model=oldModel)
                        # Get old belief state.
                        beliefs = copy.deepcopy(original)
                        # Project direct effect of the actions, including possible observations
                        logging.debug('Stepping ')
                        self.world.step({self.name: myAction} if myAction else None,beliefs,
                            keySubset=beliefs.keys(),horizon=horizon,updateBeliefs=True)
                        logging.debug('Step complete')
                        P[myAction][horizon] = copy.deepcopy(beliefs)
                    # Condition on actual observations
                    for o in self.omega:
                        if o not in beliefs:
                            raise ValueError('Observable variable %s missing from beliefs of %s' % (o,self.name))
                        value = vector[o]
                        for b in beliefs.distributions[beliefs.keyMap[o]].domain():
                            if b[o] == value:
                                break
                        else:
                            if o == oldModelKey:
                                continue
                            else:
                                newModel = None
                                logging.warning('%s (model %s) has impossible observation %s=%s when doing %s' % \
                                              (self.name,oldModel,o,self.world.float2value(o,vector[o]),myAction))
                                logging.warning('Allowable values are: {}'.format(', '.join([str(t) for t in beliefs.marginal(o).domain()])))
                                if o in self.world.dynamics and myAction in self.world.dynamics[o]:
                                    logging.warning('Action effect is:\n%s' % (self.world.dynamics[o][myAction]))
                                    if original is not None:
                                        logging.warning('Believed values are:\n%s' % ('\n'.join(['\t%s: %s' % (k,self.world.getFeature(k,original))
                                            for k in self.world.dynamics[o][myAction].getKeysIn() if k !=CONSTANT])))
                                    logging.warning('Original values are:\n%s' % ('\n'.join(['\t%s: %s (%d)' % (k,self.world.getFeature(k,vector),vector[k])
                                        for k in self.world.dynamics[o][myAction].getKeysIn() if k !=CONSTANT and k in vector])))
                                break
                        beliefs[o] = vector[o]
                    else:
                        # Create model with these new beliefs
                        # TODO: Look for matching model?
                        for dist in beliefs.distributions.values():
                            if len(dist) > 1:
                                deletion = False
                                for vec in dist.domain():
                                    if dist[vec] < self.epsilon:
                                        del dist[vec]
                                        deletion = True
                                if deletion:
                                    dist.normalize()
                        newModel = self.belief2model(oldModel,beliefs)['name']
                        SE[omega][myAction][horizon] = newModel
                        if oldModelKey in self.omega:
                            # Observe this new model
                            self.world.setFeature(oldModelKey,newModel,beliefs)
            # Insert new model into true state
            if isinstance(newModel,str):
                vector[newModelKey] = self.world.value2float(oldModelKey,newModel)
                newDist.addProb(vector,prob)
            elif newModel is not None:
                raise RuntimeError('Unable to process stochastic belief updates: %s' % (newModel))
        assert len(newDist) > 0,'Impossible observations'
        newDist.normalize()
        change = False
        for vec in newDist.domain():
            if self.belief_threshold is not None and newDist[vec] < self.belief_threshold:
                del newDist[vec]
                change = True
        if change:
            assert len(newDist) > 0
            newDist.normalize()
        return trueState

class ValueFunction:
    """
    Representation of an agent's value function, either from caching or explicit solution
    """
    def __init__(self,xml=None):
        self.table = []
        if xml:
            self.parse(xml)

    def get(self,name,state,action,horizon,ignore=None):
        try:
            V = self.table[horizon]
        except IndexError:
            return None
        if V:
            if ignore:
                substate = state.filter(ignore)
                if substate in V:
                    value = V[substate][name][action]
                else:
                    substate = self.world.nearestVector(substate,V.keys())
                    value = V[substate][name][action]
                return value
            else:
                try:
                    value = V[state][name][action]
                    return value
                except KeyError:
                    pass
        return None

    def set(self,name,state,action,horizon,value):
        while True:
            try:
                V = self.table[horizon]
                break
            except IndexError:
                self.table.append({})
        if not state in V:
            V[state] = {}
        if not name in V[state]:
            V[state][name] = {}
        V[state][name][action] = value

    def add(self,name,state,action,horizon,value):
        """
        Adds the given value to the current value function
        """
        previous = self.get(name,state,action,horizon)
        if previous is None:
            # No previous value, take it to be 0
            self.set(name,state,action,horizon,value)
        else:
            # Add given value to previous value
            self.set(name,state,action,horizon,previous+value)

    def actionTable(self,name,state,horizon):
        """
        
    :returns: a table of values for actions for the given agent in the given state
        """
        V = self.table[horizon]
        table = dict(V[state][name])
        if None in table:
            del table[None]
        return table

    def printV(self,agent,horizon):
        V = self.table[horizon]
        for state in V.keys():
            print
            agent.world.printVector(state)
            print(self.get(agent.name,state,None,horizon))

    def __lt__(self,other):
        return self.name < other.name

def explain_decision(decision):
    print(decision.keys())