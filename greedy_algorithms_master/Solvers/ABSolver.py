#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" ABSolver.py
Abstract class for solvers. The implemented algorithms share the common structure of 
iteratively including and excluding candidate features with various rules.
 
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

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import copy


@dataclass
class Feature:
    coef: np.array
    ID: int


class ABSolver(ABC):
    """Abstract class of solvers"""

    def __init__(self):
        self.blockID = []
        self.maxIter = 20
        self.shrinkBy = 0
        self.expandBy = 0

    def Solve(self, A, b, sparsity, normalize=False, verbose=False):
        self.blockID = []
        self.sparsity = sparsity
        self.verbose = verbose
        if normalize:
            self.A = A.GetNormalized()
        else:
            self.A = A
        self.b = copy.copy(b)
        self.residual = copy.copy(b)
        self._Initialize()
        if self.verbose:
            print("Initialization - residual: ", self.residual)

        for _ in range(self.maxIter):
            self._Expand()
            self._Shrink()
            if self._Terminate():
                break

        coef = np.linalg.lstsq(self.A.GetSubMatrix(self.blockID), self.b, rcond=None)[0]
        blockCoef = []
        ii = 0
        for n in range(self.A.groupNumber):
            if n in self.blockID:
                blockCoef.append(coef[ii * self.A.groupSize:(ii + 1) * self.A.groupSize])
                ii = ii + 1
            else:
                blockCoef.append(np.zeros((self.A.groupSize,)))
        return self.blockID, blockCoef

    @abstractmethod
    def _Initialize(self):
        pass

    @abstractmethod
    def _GetExpansionScores(self):
        pass

    @abstractmethod
    def _GetShrinkingScores(self):
        pass

    def _Expand(self):
        scoreList = self._GetExpansionScores()
        if self.verbose:
            print("Expansion score: ", scoreList)
        # argsort is ascending
        newSet = np.argsort(scoreList)[-self.expandBy:]
        if len(self.blockID) == 0:
            self.blockID = newSet
        else:
            self.blockID = np.union1d(self.blockID, newSet)
        self.blockID = np.sort(self.blockID)

    def _Shrink(self):
        if self.shrinkBy == 0:
            return
        scoreList = self._GetShrinkingScores()
        if self.verbose:
            print("Shrinking score: ", scoreList)
        newSet = np.argsort(scoreList)[-self.shrinkBy:]
        self.blockID = np.sort(self.blockID[newSet])

    @abstractmethod
    def _Terminate(self):
        pass

    def _proj(self, y, A):
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        proj = A @ b

        return proj

    def _resid(self, y, A):
        return y - self._proj(y, A)
