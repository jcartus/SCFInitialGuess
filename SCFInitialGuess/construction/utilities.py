"""Consists of all the utilites used in construction of matrices. 

Author:
- Johannes Cartus, TU Graz
"""

import numpy as np

from SCFInitialGuess.utilities.constants import \
        number_of_basis_functions as N_BASIS


def embed(x, y, mask):
    """Embed a square matrix x with y where mask is true.

    Args:
        x <np.array>: to be embedded matrix
        y <np.array>: elements that are embedded into x. Same size as x.
        mask <np.array<bool>>: marks where to embed. Same size as x and y.
    """

    p = x.copy()
    p[mask] = (y.copy())[mask]
    return p

def embed_batch(X, Y, mask):
    """Embed a square matrix x with y where mask is true.

    Args:
        x <list<np.array>>: set of to be embedded matrices
        y <list<np.array>>: set of elements that are embedded into elements of X. 
            Same size as X.
        mask <np.array<bool>>: marks where to embed. 
    """

    f_embedded = []
    for (x, y) in zip(X, Y):
        f_embedded.append(embed(x, y, mask))
    return np.array(f_embedded)


def make_center_mask(mol):
    """Create a boolean matrix that is true for center block elements, and 
    false else.

    mol <SCFInitialGuess.utilities.dataset.Molecule>: molecule that determines 
        basis set and composition.
    """
    
    dim = mol.dim

    mask = np.zeros((dim, dim))

    current_dim = 0
    for atom in mol.species:
        
        # calculate block range
        index_start = current_dim
        current_dim += N_BASIS[mol.basis][atom] 
        index_end = current_dim
        
        # calculate logical vector
        L = np.arange(dim)
        L = np.logical_and(index_start <= L, L < index_end)
        
        m = np.logical_and.outer(L, L)
        mask = np.logical_or(mask, m)

    return mask

def make_homo_mask(mol):
    """Create a boolean matrix that is true for all off-diagonal overlaps 
    of atoms that are of the same element.
    
    mol <SCFInitialGuess.utilities.dataset.Molecule>: molecule that determines 
        basis set and composition.
    """

    dim = mol.dim


    mask = np.zeros((dim, dim))

    current_dim_i = 0
    for i, atom_i in enumerate(mol.species):
        
        
        # calculate block range
        index_start_i = current_dim_i
        current_dim_i += N_BASIS[mol.basis][atom_i] 
        index_end_i = current_dim_i
        
        # calculate logical vector
        L_i = np.arange(dim)
        L_i = np.logical_and(index_start_i <= L_i, L_i < index_end_i)
        
        current_dim_j = 0
        for j, atom_j in enumerate(mol.species):
            
            # calculate block range
            index_start_j = current_dim_j
            current_dim_j += N_BASIS[mol.basis][atom_j] 
            index_end_j = current_dim_j

            if i == j:
                continue
            
            if atom_i == atom_j:
                # calculate logical vector
                L_j = np.arange(dim)
                L_j = np.logical_and(index_start_j <= L_j, L_j < index_end_j)


                m = np.logical_and.outer(L_i, L_j)
                
                mask = np.logical_or(mask, m)
        
    return mask

def make_hetero_mask(mol):
    """Create a boolean matrix that is true for all off-diagonal overlaps 
    of atoms that are NOT of the same element.
    
    mol <SCFInitialGuess.utilities.dataset.Molecule>: molecule that determines 
        basis set and composition.
    """

    dim = mol.dim

    mask = np.zeros((dim, dim))

    current_dim_i = 0
    for i, atom_i in enumerate(mol.species):
        
        
        # calculate block range
        index_start_i = current_dim_i
        current_dim_i += N_BASIS[mol.basis][atom_i] 
        index_end_i = current_dim_i
        
        # calculate logical vector
        L_i = np.arange(dim)
        L_i = np.logical_and(index_start_i <= L_i, L_i < index_end_i)
        
        current_dim_j = 0
        for j, atom_j in enumerate(mol.species):
            
            # calculate block range
            index_start_j = current_dim_j
            current_dim_j += N_BASIS[mol.basis][atom_j] 
            index_end_j = current_dim_j

            if i == j:
                continue
            
            if atom_i != atom_j:
                # calculate logical vector
                L_j = np.arange(dim)
                L_j = np.logical_and(index_start_j <= L_j, L_j < index_end_j)


                m = np.logical_and.outer(L_i, L_j)
                
                mask = np.logical_or(mask, m)

    return mask