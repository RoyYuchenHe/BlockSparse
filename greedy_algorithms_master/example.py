#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Example to demonstrate the usage
This program shows the general interfaces for the follwing greedy algorithms
for recoverying block sparse signals:
BOMP, BSP, BOMPR, BCoSaMP, and GPSP (See README.md for the citations)
To use GPSP, for example: solver = GPSP.Solver()
Then solver.Solve(...)

 
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

from BlockMatrix import BlockMatrix
from BlockVector import BlockVector
from Solvers import *
import numpy as np

# Define the data structure
groupNumber = 10
groupSize = 5
sparsity = 2  # k sparsity
groupMean = np.random.normal(1.0, 5.0, (groupNumber,))
sampleSize = 50
groupStd = np.abs(np.random.normal(1.0, 5.0, (groupNumber,)))
blockA = BlockMatrix.Gaussian(groupMean, groupStd, groupSize, sampleSize)
signalMean = np.random.normal(1.0, 5.0, (groupNumber,))
signalStd = np.ones((groupNumber,))
trueSignal = BlockVector.Gaussian(signalMean, signalStd, groupSize, sparsity)
b = blockA.Multiply(trueSignal)

solver = GPSP.Solver()  # Choice of solver
chosenID, estCoef = solver.Solve(blockA, b, sparsity, normalize=True, verbose=False)
print("Identified feature ID: ", chosenID)  # Default true signals have their first k blocks non-zeros.
