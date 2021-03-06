"""This file contains everyting required to generate plots.

Author:
    - Johannes Cartus, QCIEP, TU Graz
"""

from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
from pyscf.scf import hf

from SCFInitialGuess.utilities.dataset import make_matrix_batch
from SCFInitialGuess.utilities.usermessages import Messenger as msg



def statistics(x):
    return np.mean(x), np.std(x)

def matrix_error(error, xlabel="index", ylabel="index", ButadienMode=False, **kwargs):
    
    ax = sns.heatmap(error, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


    if ButadienMode:
        C_labels = ["1s  ", "2s  ", "2px", "2py", "2pz"]
        H_labels = ["1s  "]
        labels = [
            str(ci) + ". C: " + orbital \
                for ci in range(1,5) for orbital in C_labels
        ] + [
            str(hi) + ". H: " + orbital \
                for hi in range(1,7) for orbital in H_labels
        ]


        plt.yticks(np.arange(26), labels) 
        plt.xticks(np.arange(26), labels, rotation='vertical') 


    return ax
    
def prediction_scatter(
        actual,
        predicted, 
        xlabel="actual", 
        ylabel="predicted", 
        **kwargs
    ):

    data = DataFrame({xlabel: actual, ylabel: predicted})
    ax = sns.regplot(x=xlabel, y=ylabel, data=data, **kwargs)
    return ax
    
def iterations_histogram(
        dict_iterations, 
        xlabel="iterations / 1", 
        ylabel="count / 1", 
        **kargs
    ):

    data = DataFrame(dict_iterations)
    ax = sns.countplot(x=xlabel, y=ylabel, data=data)
    return ax

def plot_summary_scalars(
    file_label_dicts, 
    xlabel="steps / 1", 
    ylabel="costs / 1"
    ):
    """This function is used to plot data of scalars exported from
    tensorboard.

    Args:
        file_label_dicts <dict<str, str>>: dictionary with labels for plot and
        file path for data to be plotted.
        xlabel/ylabel <str>: plot axis labels.
    """
    
    fig = plt.figure()

    for label, fpath in file_label_dicts.items():
        with open(fpath, "r") as f:
            lines = f.readlines()[1:]

        steps, scalar = [], []
        for line in lines:
            splits = line.split(",")
            steps.append(int(splits[1]))
            scalar.append(float(splits[2]))

        plt.semilogy(steps, scalar, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()

    return fig

def density_cut_old(mol, dm, nx=80, ny=80):
    """ Calculates the density in the x-y plane on a grid"""
    
    from pyscf import lib
    from pyscf.dft import gen_grid, numint

    coords = mol.atom_coords()
    coords[:, 2] = 0 # set z-coordinate 0

    lower = np.min(coords, axis=0)
    upper = np.max(coords, axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(lower[0], upper[0], nx),
        np.linspace(lower[1], upper[1], ny)
    )
    

    grid = np.array(
        [grid_x.flatten(), 
        grid_y[:].flatten(), 
        np.zeros(grid_x.shape).flatten()]
    ).T

    ao = numint.eval_ao(mol, grid)
    rho = numint.eval_rho(mol, ao, dm)
    return rho.reshape(nx, ny)

def density(mol, dm, nx=80, ny=80, nz=80):
    from scipy.constants import physical_constants
    from pyscf import lib
    from pyscf.dft import gen_grid, numint


    coord = mol.atom_coords()
    box = np.max(coord,axis=0) - np.min(coord,axis=0) + 6
    boxorig = np.min(coord,axis=0) - 3
    xs = np.arange(nx) * (box[0]/nx)
    ys = np.arange(ny) * (box[1]/ny)
    zs = np.arange(nz) * (box[2]/nz)
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = np.asarray(coords, order='C') - (-boxorig)

    ngrids = nx * ny * nz
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx, ny, nz)

    # needed for conversion as x,y are in bohr for some reason
    a0 = physical_constants["Bohr radius"][0]

    return (
        rho.T, 
        (xs + boxorig[0]) * a0, 
        (ys + boxorig[1]) * a0, 
        (zs + boxorig[2]) * a0
    )


def density_cut(mol, dm, nx=80, ny=80, z_value=0):
    from scipy.constants import physical_constants
    from pyscf import lib
    from pyscf.dft import gen_grid, numint

    nz = 1

    coord = mol.atom_coords()
    box = np.max(coord,axis=0) - np.min(coord,axis=0) + 6
    boxorig = np.min(coord,axis=0) - 3
    xs = np.arange(nx) * (box[0]/nx)
    ys = np.arange(ny) * (box[1]/ny)
    zs = np.array([z_value]) 
    #zs = np.arange(nz) * (box[2]/nz)
    coords = lib.cartesian_prod([xs,ys,zs])
    coords = np.asarray(coords, order='C') - (-boxorig)

    ngrids = nx * ny * nz
    blksize = min(8000, ngrids)
    rho = np.empty(ngrids)
    ao = None
    for ip0, ip1 in gen_grid.prange(0, ngrids, blksize):
        ao = numint.eval_ao(mol, coords[ip0:ip1], out=ao)
        rho[ip0:ip1] = numint.eval_rho(mol, ao, dm)
    rho = rho.reshape(nx,ny)

    # needed for conversion as x,y are in bohr for some reason
    a0 = physical_constants["Bohr radius"][0]

    return rho.T, (xs + boxorig[0]) * a0, (ys + boxorig[1]) * a0

def mf_initializer(mol):
    """Will init pyscf hf engine. With damping of 0.3 and maximum of 100 
    iterations"""
    mf = hf.RHF(mol)
    mf.diis = None
    mf.verbose = 1
    mf.max_cycle = 100
    
    return mf

def mf_initializer_damping(mol):
    """Will init pyscf hf engine. With damping of 0.3 and maximum of 100 
    iterations"""
    mf = hf.RHF(mol)
    mf.diis = None
    mf.diis_start_cycle = 1000
    mf.damp = 0.3
    mf.verbose = 1
    mf.max_cycle = 100
    
    return mf

def mf_initializer_diis(mol):
    """Will init pyscf hf engine. With damping of 0.3 and maximum of 100 
    iterations"""
    mf = hf.RHF(mol)
    mf.verbose = 1
    mf.max_cycle = 100
    
    return mf

def measure_iterations(mf_initializer, guesses, molecules):
    """For an scf engine as returned by mf_initializer
    for a list of molecules and a list of corresponding guesses the number 
    of required iterations will be returned.
    """

    iterations = []
    for i, (p, molecule) in enumerate(zip(guesses, molecules)):

        msg.info("Iteration calculation: " + str(i))

        mf = mf_initializer(molecule.get_pyscf_molecule())

        try:
            mf.kernel(dm0=p)

            iterations.append(mf.iterations)

        except Exception as ex:
            msg.warn("SCF calculation failed: " + str(ex))

            iterations.append(mf.max_cycle)

    return iterations

def measure_symmetry_error(p_batch):
    """For a list of QUADRATIC Matrices calculate symmetry""" 
    for p in p_batch:
        yield np.mean(np.abs(p - p.T))

def measure_absolute_error(p, p_dataset):
    """The absolute error between a network guess p and the testing data"""
    return np.mean(np.abs(p - p_dataset), 1)

def measure_idempotence_error(p_batch, s_batch):
    for (p, s) in zip(p_batch, s_batch):
        yield np.mean(np.abs(2 * p - reduce(np.dot, (p, s, p))))

def measure_occupance_error(p_batch, s_batch, n_electrons):
    for (p, s) in zip(p_batch, s_batch):
        yield np.abs(np.trace(np.dot(p, s)) - n_electrons)

def measure_hf_energy(p_batch, molecules):

    energies = []
    for (p, mol) in zip(p_batch, molecules):
        mf = hf.RHF(mol.get_pyscf_molecule())
        h1e = mf.get_hcore()
        veff = mf.get_veff(dm=p.astype("float64"))
        energies.append(mf.energy_tot(p.astype("float64"), h1e, veff))
    return energies

def measure_hf_energy_error(p_batch, p_dataset_batch, molecules):

    E_nn = measure_hf_energy(p_batch, molecules)
    E_ds = measure_hf_energy(p_dataset_batch, molecules)

    for (e_nn, e_ds) in zip(E_nn, E_ds):
        yield np.abs(e_nn -  e_ds)
        



def measure_all_quantities(
        p,
        dataset,
        molecules,
        n_electrons,
        mf_initializer,
        dim,
        is_triu=False,
        is_dataset_triu=None,
        s=None
    ):
    """This function calculates all important quantities of a 
    density matrix (of the dimension dim) guess p, for the testing data in dataset, 
    and molecules given in molecules with n_electron electrons in them.
    As iterations are calculated too, a function to initialize the scf engine 
    is handed over by mf_initialize.
    
    Returns:
        a tuple of tuples containing the values and error for each quantity 
        measured. Eg.g ((error1, error of error1), (error2, error of ...), ...)
    """

    if is_dataset_triu is None:
        is_dataset_triu = is_triu

    if s is None:            
        s_raw_batch = make_matrix_batch(
            dataset.inverse_input_transform(dataset.testing[0]),
            dim,
            is_dataset_triu
        )
    else:
        s_raw_batch = make_matrix_batch(s, dim, False)

    p_batch = make_matrix_batch(p, dim, is_triu)

    err_abs = statistics(list(
        measure_absolute_error(
            p_batch, 
            make_matrix_batch(dataset.testing[1], dim, is_dataset_triu)
        )
    ))

    err_sym = statistics(list(
        measure_symmetry_error(p_batch)
    ))

    err_idem = statistics(list(
        measure_idempotence_error(p_batch, s_raw_batch)
    ))

    err_occ = statistics(list(
        measure_occupance_error(p_batch, s_raw_batch, n_electrons)
    ))

    err_Ehf = statistics(list(
        measure_hf_energy_error(
            p_batch, 
            make_matrix_batch(dataset.testing[1], dim, is_dataset_triu), 
            molecules
        )
    ))

    iterations = np.array(measure_iterations(
        mf_initializer, 
        p_batch.astype('float64'), 
        molecules
    ))

    max_cycle = mf_initializer(
        molecules[0].get_pyscf_molecule()
    ).max_cycle


    return (
        err_abs, 
        err_sym, 
        err_idem, 
        err_occ, 
        err_Ehf, 
        statistics(iterations),
        statistics(iterations[iterations != max_cycle]),
        np.sum(max_cycle == np.array(iterations))
    )

def make_results_str(results):
    """Creates a printable string from results of measure all quantities"""

    out = ""

    def format_results(result):
        if isinstance(result, list):
            out = list(map(
                lambda x: "{:0.5E} +- {:0.5E}".format(*x),
                result
            ))
            out = "\n".join(out)
        else:
            out =  "{:0.5E} +- {:0.5E}".format(*result)
        return out

    out += "--- Absolute Error ---\n"
    out += format_results(results[0])
    out += "\n"
    out += "--- Symmetry Error ---\n"
    out += format_results(results[1])
    out += "\n"
    out += "--- Idempotence Error ---\n"
    out += format_results(results[2])
    out += "\n"
    out += "--- Occupance Error ---\n"
    out += format_results(results[3])
    out += "\n"
    out += "--- HF Energy Error ---\n"
    out += format_results(results[4])
    out += "\n"
    out += "--- Avg. Iterations ---\n"
    out += format_results(results[5])
    out += "\n"
    out += "--- Avg. Iterations W/O Non Converged ---\n"
    out += format_results(results[6])
    out += "\n"
    out += "--- Num. Not Convd. ---\n"
    out += str(results[7])
    out += "\n"

    return out




def analyze_raw(p, p_ref, s, mol):
    """Calculate error and properties sample by sample. Advantage: can be 
    done in a loop for samples of differnt dimension!
    
    Args:
        p <np.array>: density to be analzed
        p_ref <np.array>: converged density of the same sample
        s <np.array>: overlap matrix of molecule.
        mol <SCFInitialGuess.utilities.dataset.Molecule>: the sample
    """

    # reshape (make sure quadratic matrices)
    s_batch = s.reshape(-1, int(np.sqrt(np.prod(s.shape))))
    p_ref_batch = p_ref.reshape(s_batch.shape)
    p_batch = p.reshape(s_batch.shape)
    
    # measure
    err_abs = np.mean(np.abs(p_batch - p_ref_batch))
    err_hf = next(measure_hf_energy_error([p_batch], [p_ref_batch], [mol]))
    err_idem = next(measure_idempotence_error([p_batch], [s_batch]))
    try: 
        n = mol.n_electrons()
    except:
        from SCFInitialGuess.utilities.constants import atomic_numbers as Z
        n = np.sum([Z[atom] for atom in mol.species])
    
    err_occ = next(measure_occupance_error([p_batch], [s_batch], n))
    
    return (
        err_abs,
        err_hf,
        err_idem,
        err_occ
    )
    
    
def analyze_raw_batch(P, P_ref, S, molecules):
    """Batch version of analyze raw"""
    
    n_samples = len(P)
    
    errors = []
    for i, (p, p_ref, s, mol) in enumerate(zip(P, P_ref, S, molecules)):
        msg.info(str(i+1) + " / " + str(n_samples))
        errors.append(analyze_raw(p, p_ref, s, mol))
    
    errors = np.array(errors)
    
    return (
        statistics(errors[:,0]), # abs
        statistics(errors[:,1]), # hf
        statistics(errors[:,2]), # idem
        statistics(errors[:,3]) # occ
    )

def format_raw(results):
    """Mate a formatted string for logging/printing from results of 
    analyze_raw_batch"""

    msg = ""
    msg += "AbsError: {:0.5E} +- {:0.5E}\n".format(*(results[0]))
    msg += "EhfError: {:0.5E} +- {:0.5E}\n".format(*(results[1]))
    msg += "IdemEror: {:0.5E} +- {:0.5E}\n".format(*(results[2]))
    msg += "OccError: {:0.5E} +- {:0.5E}\n".format(*(results[3]))
    return msg