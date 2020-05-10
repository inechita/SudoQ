"""
SudoQ - software for solving quantum Sudoku
Copyright (C) 2020  Jordi Pillet, Ion Nechita

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy.linalg import eigh, svd
from math import sqrt
import glob
import random

def readSudoku(fileName):
    """
    Reads a Sudoku grid from a text file
    Empty cells are presented by 0
    Convert to internal format where empty cells are -1
    """
    
    data = np.loadtxt(fileName, skiprows=0, delimiter=" ", dtype=np.int);
    
    n = data.shape[0]
    return (data - np.ones([n, n], dtype=np.int))

def constraintListMagic(n, randomize = False):
    """
    Builds the list of constraints for a magic unitary grid of size n
    Constraints: rows, columns
    """
    
    allc = [];
    
    # add row constraints
    for i in range(n):
        c = [];
        for j in range(n):
            c.append([i,j])
        allc.append(c)

    # add column constraints
    for j in range(n):
        c = [];
        for i in range(n):
            c.append([i,j])
        allc.append(c)    
    
    # output constraint list
    return allc            
            
def constraintListSudoku(n=3):
    """
    Builds the list of constraints for a Sudoku grid of size n**2
    Constraints: rows, columns, and nxn subsquares
    """
    
    allc = [];
    
    # add row constraints
    for i in range(n**2):
        c = [];
        for j in range(n**2):
            c.append([i,j])
        allc.append(c)

    # add column constraints
    for j in range(n**2):
        c = [];
        for i in range(n**2):
            c.append([i,j])
        allc.append(c)
    
    # add subsquare constraints
    for i in range(n):
        for j in range(n):
            c = [];
            for k in range(n):
                for l in range(n):
                    c.append([i*n+k,j*n+l])
            allc.append(c)
    
    # output constraint list
    return allc

def sinkhornError(a, consList):
    """
    Computes the error function for the given target constraints
    """

    n = a.shape[0]
    s = []

    # parse all constraints    
    for c in consList:
        
        # compute sums of projections in c
        partialSum = np.zeros([n, n])
        for i in range(n):
            aij = a[c[i][0], c[i][1]]
#            print("aij = ", aij)
            partialSum = partialSum + np.outer(aij, aij.conj())
            
        # compare the sum with the identity matrix    
        s.append(np.linalg.norm(partialSum - np.eye(n)))

#    print(s)      
        
    # returns the maximal norm of the differences between expected 
    # row/col sums and target values    
    return np.max(s)

def optimalTransformation(A, B):
    """
    OptimalTransformation(A, B) computes the optimal F s.t. FAF^* = B
    The function takes as input two PSD nxn
    matrices of rank r, and outputs a transformation F s.t. FAF^* = B and
    ||F-id||_2 minimal on the support of A
    """
    
    n = A.shape[0]
    r = int(round(np.trace(B)))

    DA, UA = eigh(A);
    DB, UB = eigh(B);           
    
    # truncate everything to last r coordinates (largest eigs)
    DA = DA[n-r:n]
    UA = UA[0:n, n-r:n]
    DB = DB[n-r:n]
    UB = UB[0:n, n-r:n]
    
    
    try:  
        # Dratio is the matrix with sqrt(D_B/D_A) on the diagonal
        Dratio = np.zeros(r);
        for i in range(r):    
                Dratio[i] = sqrt(DB[i] / DA[i])
   
        # take W to be the polar part of UB'*UA. W minimizes the norm of F-I
        U, _, V = svd(UB.T.conj() @ UA @ np.diag(Dratio))
        V = V.T.conj()
        W = U @ V.T.conj();
        
        return UB @ W @ np.diag(Dratio) @ UA.T.conj()
    except:
        return np.eye(n)
    
def sinkhornBalance(x, A, consList, precision = 10**(-6), maxIterations = 1000, strength = 0.5, verbose = False):
    """
    Balances a matrix of vectors in order to realize a set of contraints
    The Sinkhorn algorithm runs untill all constraints are satisfied up to 
    the given precision or unitl the maximum number of iterations is reached.
    The parameter strength dictates the smoothness of the change in 
    each iteration. 
    The function returns the grid, as well as the number of iterations
    """
    
    it = 0
    n = x.shape[0]
    
    error = sinkhornError(x, consList)
    # main loop
    while it < maxIterations and error > precision:
        
        # randomize the list of constraints at each iteration
#        newConsList = np.random.permutation(consList)  
        
        
        for c in consList:
            
            # compute the sum over the elements in the constraint c
            # S is the sum of the free elements (not present in the grid)
            S = np.zeros([n, n])
            # T is the sum of the fixed elemens (present in the grid)
            T = np.zeros([n, n])
            
            for i in range(n):
                if A[c[i][0], c[i][1]] < 0:
                    xi = x[c[i][0], c[i][1]]
                    S = S + np.outer(xi, xi.conj())
                else:
                    xi = np.zeros([n, 1])
                    xi[A[c[i][0], c[i][1]]] = 1.0
                    T = T + np.outer(xi, xi.conj())
            
            # if there is something to be done (i.e. S is non-empty)
            if not np.allclose(S, np.zeros([n, n])):
                # take the complementary subspace for T
                T = np.eye(n) - T;           
                   
                # compute the optimal transofrmation
                R = optimalTransformation(S, T)

                # interpolate between the square inverse and the identity matrix
                # using the strength parameter
                R = strength*R + (1-strength)*np.eye(n)

                # normalize eleements in c
                for i in range(n):
                    if A[c[i][0], c[i][1]] < 0:
                        xi = x[c[i][0], c[i][1]]
                        x[c[i][0], c[i][1]] = R @ xi
                        
                SS = np.zeros([n, n])       
                for i in range(n):
                    xi = x[c[i][0], c[i][1]]
                    SS = SS + np.outer(xi, xi.conj())        

        error = sinkhornError(x, consList)
        it = it + 1
        
        if verbose:
            print(it, " --- ", error)
            
    return x, it

def resetGrid(x, A):
    """
    Resets the positions in the matrix of vectors x to the correct values 
    given by the matrix A. N
    """
    
    n = x.shape[0]
    
    for i in range(n):
        for j in range(n):
            if (A[i,j]+1)>0:
                # set the [i, j] element to be equal to the corresponding basis vector
                x[i,j] = np.zeros(n)
                x[i, j, A[i, j]] = 1.0
                
    return x

def solveGrid(A, precision = 10**(-6), maxIterations = 500, strength = 0.5, verbose = False):

    
    n = A.shape[0]   
    
    # initialize with random Gaussian vectors
    x = np.random.randn(n, n, n) + 1j * np.random.randn(n, n, n)

    # load Sudoku constraint list
    consList = constraintListSudoku(int(np.sqrt(n)))
    

    # sets the vectros corresponding to filled positions to the correct value
    x = resetGrid(x, A)

    return sinkhornBalance(x, A, consList, precision, maxIterations, strength)

def isCommutative(a):
    """
    Returns a measure of the non-commutativity of a QLS: if the QLS is classical
    the returned value is very small; if not, it rereturns the value of the 
    squared absolute value of a scalar product the farthest away from 0 and 1
    as well as the matrix indices where this occurs
    """
    n = a.shape[0]
    
    maxsp = -1
    best = []
    
    for i1 in range(n):
        for j1 in range(n):
            for i2 in range(i1,n):
                for j2 in range(j1, n):
                    sp = abs(np.inner(a[i1, j1], a[i2, j2].conj()))**2
#                    print(sp)
                    sp = np.min([sp, 1-sp])
                    if sp > maxsp:
                        maxsp = sp
                        best = [i1, j1, i2, j2]
                    
    return [maxsp, best]

def guessClassicalGrid(x):
    """
    Returns the closest classical grid in the computational basis to x, as 
    well as the error of the classical approximation
    """
    
    n = x.shape[0]
    
    table = np.zeros([n,n], dtype=np.int)
    
    err = 0
    
    for i in range(n):
        for j in range(n):
            
            # the candidate matrix is the argmax of the amplitudes
            posMax = 0
            for k in range(1,n):
                if abs(x[i,j,k]) > abs(x[i,j,posMax]):
                    posMax = k
                    
            table[i,j] = posMax
            err = max(err, abs(1-abs(x[i,j,posMax])))
                  
    return table, err

def violatedConstraints(x, cons, precision = 10**(-6)):
    """
    Computes the number of violated constraints in SudoQ square

    Parameters
    ----------
    x : 3D ndarray, complex
        A SudoQ square.
    cons : a list if lists of pairs
        The list of contraints we want to check.
    precision : float, optional
        The precision used to check if constraints are satisfied.
        The default is 10**(-6).

    Returns
    -------
    v : int
        The number of constraints which are NOT satisfied.

    """
    
    v = 0
    n = x.shape[0]
    
    for c in cons:
        S = np.zeros([n, n])

        for i in range(n):
            xi = x[c[i][0], c[i][1]]
            S = S + np.outer(xi, xi.conj())
            
        if np.linalg.norm(S-np.eye(n)) > precision:
            v = v + 1
    
    return v

def testStrengths(grid, repeats = 100, maxIterations = 500, 
                  strengths = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1],
                  fileName = "out.txt"):
    """
    Runs the alogrithm several times and saves the relevant output information

    Parameters
    ----------
    grid : 2D ndarray
        The Sudoku grid.
    repeats : int, optional
        The number of times the algorithm is ran for each value of the 
        strength. The default is 100.
    maxIterations : int, optional
        The maximal number of iterations for each run. The default is 500.
    strenths : int array, optional
        The array containing the strengths to be tested. The default is 
        .1 to 1 with .1 steps.
    fileName : string, optional
        The name of output file. The default is "out.txt".

    Returns
    -------
    Writes in the output file a line for each run, containing the value of the 
    strength parameter, the number of iterations the program used,
    the number of the violated contraints (=0 if the grid was solved),
    and the commutativity parameter (=~ 0 if the solution is classical).

    """
   
    d = int(sqrt(grid.shape[0]))
    
    for strength in strengths:
        for i in range(repeats):
            f = open(fileName,"a+")
            x, it = solveGrid(grid, maxIterations = maxIterations, strength = strength);
            vc = (violatedConstraints(x, constraintListSudoku(d)))
            ic = isCommutative(x)[0]
            f.write("%0.3f %d %d %0.3f\n" % (strength, it, vc, ic))
            f.close()   
            
            
def testClassicalSolution(folder, repeats = 100, outFileName = "out.txt"):
    """
    Test whether every grid in the given folder admits purely quantum solutions
    
    For every grid, output the number of iterations of the algorithm, and the 
    error of the best classical approximation

    Parameters
    ----------
    folder : string
        The folder containing the grids.
    repeats : int, optional
        The number of times the algorithm is ran for each value of the 
        strength. The default is 100.
    outFileName : string, optional
        The name of output file. The default is "out.txt".

    Returns
    -------
    Writes in the output file a line for each run, containing the number of 
    iterations, the Sinkhorn error, 
    and the error of the best classical approximation 
    (=~ 0 if the solution is classical).

    """    

    f = open(outFileName,"a+")    

    for filename in glob.glob(folder+"\\*.txt"):
        x = readSudoku(filename)
        n = int(sqrt(x.shape[0]))
        for i in range(repeats):
            y,it = solveGrid(x);
            sErr = sinkhornError(y, constraintListSudoku(n))
            z, cErr = guessClassicalGrid(y)
            f.write("%s %d %0.3f %0.3f\n" % (filename, it, sErr, cErr))
        
    f.close()
    
def eraseRandomClue(x):
    
    n = x.shape[0]
    
    # cound the number of clues present in the grid
    nClues = 0;
    for i in range(n):
        for j in range(n):
            if x[i,j] >= 0:
                nClues = nClues + 1
    
    # index of the clue to be deleted
    clue = random.randint(0, nClues-1)
    print(nClues, clue)
    
    # delete the clue
    c = 0;
    for i in range(n):
        for j in range(n):
            if x[i,j] >= 0:
                # delete the clue and return x
                if c == clue:
                    y = x
                    y[i,j] = -1;
                    return y
                else:
                    c = c + 1
   
def testClassicalSolutionAfterDeletion(folder, repeats = 100, outFileName = "out.txt"):
    """
    Test whether every grid in the given folder admits purely quantum solutions
    
    For every grid, output the number of iterations of the algorithm, and the 
    error of the best classical approximation

    Parameters
    ----------
    folder : string
        The folder containing the grids.
    repeats : int, optional
        The number of times the algorithm is ran for each value of the 
        strength. The default is 100.
    outFileName : string, optional
        The name of output file. The default is "out.txt".

    Returns
    -------
    Writes in the output file a line for each run, containing the number of 
    iterations, the Sinkhorn error, 
    and the error of the best classical approximation 
    (=~ 0 if the solution is classical).

    """    

    f = open(outFileName,"a+")    

    for filename in glob.glob(folder+"\\*.txt"):
        for i in range(repeats):
            x = readSudoku(filename)
            n = int(sqrt(x.shape[0]))
            y,it = solveGrid(eraseRandomClue(x));
            sErr = sinkhornError(y, constraintListSudoku(n))
            z, cErr = guessClassicalGrid(y)
            f.write("%s %d %0.3f %0.3f\n" % (filename, it, sErr, cErr))
        
    f.close()