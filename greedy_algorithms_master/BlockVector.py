#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Block vector class
A basic implementation for the structure of a block vector with necessary methods
 
This file is part of the BlockSparse distribution (https://github.com/RoyYuchenHe/BlockSparse).
Copyright (c) 2024 Roy Yuchen He.
 
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Roy Yuchen He"
__contact__ = "royhe2@cityu.edu.hk"
__copyright__ = "Copyright 2024, Roy Yuchen He"
__date__ = "2024/06/01"
__deprecated__ = False
__email__ = "royhe2@cityu.edu.hk"
__license__ = "GPLv3"
__maintainer__ = "Roy Yuchen He"
__status__ = "Production"
__version__ = "0.0.1"

import numpy as np


class BlockVector(object):
    """Block vector class"""

    def __init__(self, vectorBlocks):
        self.blockList = [block for block in vectorBlocks]
        self.groupNumber = len(self.blockList)
        self.groupSize = len(self.blockList[0])

    @classmethod
    def Gaussian(cls, meanList, stdList, groupSize, sparsity):
        """Initialize blockList by random Gaussian"""
        blocks = []
        for n in range(len(meanList)):
            if n < sparsity:
                mean = meanList[n]
                std = stdList[n]
                blocks.append(np.random.normal(mean, std, size=(groupSize,)))
            else:
                blocks.append(np.zeros((groupSize,)))

        return cls(blocks)

    @classmethod
    def Poisson(cls, meanList, groupSize, sparsity):
        """Initialize blockList by random Poisson"""
        blocks = []
        for n in range(len(meanList)):
            if n < sparsity:
                mean = meanList[n]
                blocks.append(np.random.poisson(mean, size=(groupSize,)))
            else:
                blocks.append(np.zeros((groupSize,)))
        return cls(blocks)

    def vector(self):
        return np.hstack([block for block in self.blockList])
