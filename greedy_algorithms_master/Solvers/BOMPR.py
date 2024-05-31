#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" BOMPR.py
Implementation of the algorithm
Fu, Y., Li, H., Zhang, Q. and Zou, J., 2014.
Block-sparse recovery via redundant block OMP. Signal Processing, 97, pp.162-171.

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

from .ABSolver import ABSolver
import numpy as np


class Solver(ABSolver):
    def __init__(self):
        super().__init__()

    def _Initialize(self):
        self.expandBy = 1
        self.shrinkBy = 0
        self.maxIter = self.sparsity

    def _GetExpansionScores(self):
        return [-np.linalg.norm(self._resid(self.residual, block), ord=2) for block in self.A.blockList]

    def _GetShrinkingScores(self):
        pass

    def _Terminate(self):
        subMat = self.A.GetSubMatrix(self.blockID)
        self.residual = self._resid(self.b, subMat)
        return False
