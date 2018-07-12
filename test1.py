#!/usr/bin/env python

import os, sys

from env.macros import *
from env.gworld import GridWorld
from env.visualize import Visualize

class Episode():
	def __init__(self,
		h=9, w=9,
		nagents_rand=4,
		boundWalls=True,
		maxlen = 1000):
		