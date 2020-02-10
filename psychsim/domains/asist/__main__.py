from argparse import ArgumentParser
import logging

from psychsim.pwl import *
from psychsim.world import World
from psychsim.agent import Agent


if __name__ == '__main__':
    # Command-line arguments
    parser = ArgumentParser()
    parser.add_argument('-d','--debug',default='WARNING',help='Level of logging detail')
    args = vars(parser.parse_args())
    level = getattr(logging, args['debug'].upper(), None)
    if not isinstance(level, int):
        raise ValueError('Invalid debug level: %s' % args['debug'])
    logging.basicConfig(level=level)

    # Create the overall environment
    world = World(single=True)

    # Create one victim
    victim = world.addAgent('Victim 1')

    world.defineState(victim.name,'status',list,['unsaved','saved','dead'])
    victim.setState('status','unsaved')

    world.defineState(victim.name,'health',float,description='How far away this victim is from dying')
    victim.setState('health',1)

    world.defineState(victim.name,'value',int,description='Value earned by saving this victim')
    victim.setState('value',5)

    world.defineState(victim.name,'location',int,description='Room number where victim is')
    victim.setState('location',3)

    # Create the player
    player = world.addAgent('Player 1')

    world.defineState(player.name,'location',int,description='Room number where player is')
    player.setState('location',0)

    # Player can move, but only one room up
    location = stateKey(player.name,'location')
    moves = {} # Save the move objects for easier access later
    for room in range(4):
        tree = makeTree({'if': equalRow(location,{(room-1)%4,(room+1)%4}),
            True: True, False: False})
        moves[room] = player.addAction({'verb': 'moveTo', 'object': '%d' % (room)},tree)
        tree = makeTree(setToConstantMatrix(location,room))
        world.setDynamics(location,moves[room],tree)

    # Player can save, but only if same room as victim
    tree = makeTree({'if': equalFeatureRow(location,stateKey(victim.name,'location')),
        True: True, False: False})
    save = player.addAction({'verb': 'save','object': victim.name},tree)
    tree = makeTree(setToConstantMatrix(stateKey(victim.name,'status'),'saved'))
    world.setDynamics(stateKey(victim.name,'status'),save,tree)

    # Pop quiz: 
    # Q: What can the agent do now?
    # A: Move to rooms 1 or 3
    legal = player.getActions()
    assert len(legal) == 2
    for action in legal:
        assert action['verb'] == 'moveTo'
        assert int(action['object']) in {1,3}

    # Player goals
    goal = makeTree({'if': equalRow(stateKey(victim.name,'status'),'saved'),
        True: setToFeatureMatrix(rewardKey(player.name),stateKey(victim.name,'value')),
        False: setToConstantMatrix(rewardKey(player.name),0)})
    player.setReward(goal,1)

    # Player lookahead
    player.setAttribute('horizon',5)

    world.setOrder([{player.name}])

    world.printState()
    print(player.getActions())
    # The player moves to room 1
    result = world.step(moves[1])
    world.printState()

    # The player decides autonomously
    print(player.getActions())
    world.step()
    world.printState()

    print(player.getActions())
    world.step()
    world.printState()
