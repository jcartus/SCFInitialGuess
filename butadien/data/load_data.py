from .grep_routines import grep_sp
import numpy as np
from os.path import join

from SCFInitialGuess.utilities.usermessages import Messenger as msg

def load_data(path, num=200):

	# Repulsive energy
	e_rep_list = np.zeros(num)
	# Overlap matrices
	overlap_list = []
	# Core Hamiltonian
	core_list = []
	# Fock matrices
	fock_list = []
	# MO energies
	e_mos_list = []
	# Convergeed densities
	density_list = []
	JK_list = []
	# Total energies using STO3G
	energies_3G = np.zeros(num)
	# List of Mulliken charges
	mull_list = []

	for i in range(num):
		msg.info(str(i + 1) + "/" + str(num))
		f = join(path, '{}.out'.format(i))
		(e_rep,overlap,core,fock,e_mos,density,energy,mull) = grep_sp(f)
		#e_rep_list[i] = e_rep[0]
		overlap_list.append(overlap[0].reshape(len(density[0])**2, ))
		core_list.append(core[0].reshape(len(density[0])**2, ))
		fock_list.append(fock[0].reshape(len(density[0])**2, ))
		#e_mos_list.append(e_mos[0])
		density_list.append(density[0].reshape(len(density[0])**2, ))
		#JK_list.append(sum(density[0]*core[0]))
		#energies_3G[i] = energy
		#mull_list.append(mull)

	return np.array(overlap_list), np.array(density_list)

if __name__ == '__main__':
	load_data("data/data")
