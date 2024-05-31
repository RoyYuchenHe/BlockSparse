#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Block matrix class
A basic implementation for the structure of a block matrix with necessary methods
 
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


class BlockMatrix(object):
    """Block matrix class"""
    def __init__(self, matrixBlocks):
        self.blockList = [block for block in matrixBlocks]
        self.groupNumber = len(self.blockList)
        self.groupSize = self.blockList[0].shape[1]

    @classmethod
    def Gaussian(cls, meanList, stdList, groupSize, observSize):
        """Initialize blockList by random Gaussian"""
        blocks = [np.random.normal(mean, stdList[n], size=(observSize, groupSize))
                  for n, mean in enumerate(meanList)]
        return cls(blocks)

    @classmethod
    def CovariateGaussian(cls, meanList, groupNumber, groupSize, observSize):
        """Initialize blockList by covariant random Gaussian"""
        L = np.tril(np.random.rand(groupNumber, groupNumber))
        cov = np.dot(L, L.T)
        A = np.random.multivariate_normal(meanList, cov, (observSize, groupSize))
        blocks = [A[..., n] for n in range(groupNumber)]
        return cls(blocks)

    @classmethod
    def Binary(cls, groupNumber, groupSize, observSize):
        """Initialize blockList by Bernoulli"""
        blocks = [np.random.randint(2, size=(observSize, groupSize)) for _ in range(groupNumber)]
        return cls(blocks)

    @classmethod
    def Poisson(cls, meanList, groupSize, observSize):
        """Initialize blockList by Poisson"""
        blocks = [np.random.poisson(lam=mean, size=(observSize, groupSize))
                  for mean in meanList]
        return cls(blocks)

    def GetNormalized(self):
        out = [block / (np.linalg.norm(block, axis=0) + 1e-16) for block in self.blockList]
        return BlockMatrix(out)

    def GetSubMatrix(self, blockIDs):
        return np.hstack([self.blockList[n] for n in blockIDs])

    def matrix(self):
        return np.hstack([block for block in self.blockList])

    def Multiply(self, b):
        return sum([mat @ vec for mat, vec in zip(self.blockList, b.blockList)])
