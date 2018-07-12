#!/usr/bin/env python3

import numpy as np
import scipy.signal
from env.macros import *
from env.visualize import *
import random


enable_visualization = True
try:
    from Tkinter import *
except ImportError:
    try:
        from tkinter import *
    except ImportError:
        print('Unable to import Tkinter. Animation will be disabled.')
        enable_visualization = False

class GridWorld:

    def __init__(self, h, w, rocks=None, rand_nagents=0, boundwalls=True):
        self.h = h
        self.w = w
        self.cells = np.zeros((h, w), dtype=float)
        self.visualize = None
        self.add_rocks(rocks)
        self.aindx_cpos = dict()
        if(boundwalls):
            self.add_rocks(self.get_boundwalls())
            if(rand_nagents > 0):
                self.add_agents_rand(rand_nagents)

    def xy_saturate(self, x, y):
        if(x<0): x=0
        if(x>self.w-1): x=self.w-1
        if(y<0): y=0
        if(y>self.h-1): y=self.h-1
        return(x, y)

    def get_boundwalls(self):
        h, w = self.get_size()
        bwalls = set()
        for x in range(w):
            for val in range(SENSE_RANGE):
                bwalls.add((val, x))
                bwalls.add((h - val - 1, x))
        for y in range(h):
            for val in range(SENSE_RANGE):
                bwalls.add((y, val))
                bwalls.add((y, w - val - 1))
        return tuple(bwalls)

    def add_rocks(self, rocks):
        if rocks:
            for rock in rocks:
                rockx, rocky = self.xy_saturate(rock[1], rock[0])
                if( not self.is_blocked(rocky, rockx) ):
                    self.cells[rocky][rockx] = IS_ROCK

    def add_agents(self, agents_spos):
        if agents_spos:
            print('Start pos: ', agents_spos)
            # Replace list of tuples with a dict lookup for better performance
            for (sy, sx) in agents_spos:
                nagents = len( self.aindx_cpos.keys() )
                if(not self.is_blocked(sy, sx)):
                    if(self.cells[sy][sx] == UNOCCUPIED):
                        aindx = nagents + 1
                        self.aindx_cpos[aindx] = (sy, sx)
                        self.cells[sy][sx] = aindx
                    else:
                        raise Exception('Cell has already been occupied!')
                else:
                    raise Exception( 'Failure! agent index: ' + str(nagents + 1) )
                    return False
            return True
        return False

    def add_agents_rand(self, nagents = 0):
        if(nagents):
            maxy, maxx = self.h - 1, self.w - 1
            agent_pos = set()
            while len(agent_pos) < nagents:
                y = random.randint(0, maxy)
                x = random.randint(0, maxx)
                if( self.passable( (y,x) ) ):
                    agent_pos.add( (y,x) )
            self.add_agents(agent_pos)

    def pos_to_action(self, cpos, npos):
        cy, cx = cpos[0], cpos[1]
        ty, tx = npos[0], npos[1]
        if (self.is_blocked(ty, tx)):
            raise Exception('Npos is blocked/invalid!')
        if(tx - cx == 1): action = Actions.RIGHT
        elif(tx - cx == -1): action = Actions.LEFT
        elif(ty - cy == 1): action = Actions.DOWN
        elif(ty - cy == -1): action = Actions.UP
        else: action = Actions.WAIT
        return action

    def is_validpos(self, y, x):
        if x < 0 or x > self.w - 1 or y < 0 or y > self.h - 1:
            return False
        else:
            return True
    def is_blocked(self, y, x):
        # print('Cell :', y, x)
        if not self.is_validpos(y, x): return True
        if(self.cells[y][x] == IS_ROCK): return True
        return False

    def passable(self, cell, constraints = None):
        y, x = cell[0], cell[1]
        if( self.is_blocked(y,x) ):
            return False
        elif( self.cells[y][x] != UNOCCUPIED ):
            return  False
        else:
            return True

    def get_size(self):
        return (self.h, self.w)

    def get_agents(self):
        return self.aindx_cpos.keys()

    def get_aindx_from_pos(self, pos):
        y, x = pos[0], pos[1]
        if( self.is_validpos(y, x) ):
            return self.cells[y][x]
        else:
            return INVALID

    @staticmethod
    def quad_to_box(y, x, quadrant):
        if( quadrant == Observe.Quadrant1 ):
            y1, x1, y2, x2 = y - SENSE_RANGE, x, y + 1, x + SENSE_RANGE + 1
        elif( quadrant == Observe.Quadrant2 ):
            y1, x1, y2, x2 = y - SENSE_RANGE, x - SENSE_RANGE, y + 1, x + 1
        elif( quadrant == Observe.Quadrant3 ):
            y1, x1, y2, x2 = y, x - SENSE_RANGE, y + SENSE_RANGE + 1, x + 1
        else:
            y1, x1, y2, x2 = y, x, y + SENSE_RANGE + 1, x + SENSE_RANGE + 1
        return y1, x1, y2, x2

    def get_nbors_occ_array(self, y, x):
        '''
        Return contents of neighbors of given cell
        return: array [ RIGHT, UP, LEFT, DOWN, WAIT ]
        '''
        nbors = np.ones(5, dtype = int ) * INVALID
        # x, y = self.xy_saturate(x, y)
        if(x > 0):
            nbors[Actions.LEFT] = self.cells[y][x-1]
        if(x < self.w - 1):
            nbors[Actions.RIGHT] = self.cells[y][x+1]
        if(y > 0):
            nbors[Actions.UP] = self.cells[y-1][x]
        if(y < self.h - 1):
            nbors[Actions.DOWN] = self.cells[y+1][x]
        nbors[Actions.WAIT] = self.cells[y][x]
        return nbors

    def observe(self, aindx, quadrant):
        '''
        Return occupancy matrix as observed by agent 'aindx'
        in quadrant 'quadrant' from its current position
        '''
        y, x = self.aindx_cpos[aindx]
        y1, x1, y2, x2 = GridWorld.quad_to_box(y, x, quadrant)
        obs_mat = self.cells[y1:y2, x1:x2].copy()
        return ( obs_mat, (y1, x1, y2, x2) )

    def action(self, aindx, action):
        retval = RWD_STEP_DEFAULT
        if(aindx in self.aindx_cpos):
            y, x = self.aindx_cpos[aindx]
        else:
            raise Exception('Agent ' + str(aindx) + ' does not exist!')
        oy, ox = y, x
        nbors = self.get_nbors_occ_array(y, x)
        if(nbors[action] == UNOCCUPIED):
        # if(nbors[action] != IS_ROCK and nbors[action] != INVALID):
            y += int(action == Actions.DOWN) - int(action == Actions.UP)
            x += int(action == Actions.RIGHT) - int(action == Actions.LEFT)
            self.aindx_cpos[aindx] = (y, x)
            self.cells[oy][ox] = 0
            self.cells[y][x] = aindx
            if(self.visualize): self.visualize.update_agent_vis(aindx)
        elif(action == Actions.WAIT):
            pass
        else:
            # raise Exception('Cell is not unoccupied! : (' + str(y) + ',' + str(x) + ') --> ' + str(action) )
            retval = RWD_BUMP_INTO_WALL

    def get_state(self, aindx):
        