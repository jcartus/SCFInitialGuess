# Routines needed for get_scf_data, a Python script for parsing the
# ingredients for an external energy calculation 
#
# AWH 2015

import numpy as np
import sys
import os
import pyQChem as qc


def grep_nuc_rep(filename):
	# Extracts the nuclear repulsion energy from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	e_rep = []

	for line in content:
		if "Nuclear Repulsion Energy" in line:
			dummy = line.split()
			e_rep.append(float(dummy[4]))

	infile.close()
	return e_rep
      
def grep_overlap(filename):
	# Extracts the Overlap Matrix from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	block = []
	lastblock = []
	overlap_list = []

	overlap_switch = 0
	for line in content:
		if "Overlap Matrix S:" in line:
			overlap_switch = 1
		if overlap_switch==1 and '.' in line and "Core" not in line:
			dummy = line.split()
			if len(dummy) == 7:
				block.append(dummy[1:])
				dim = int(dummy[0]) # Will be correct when loop is through
			else:
				lastblock.append(dummy[1:])
		if "Core" in line and overlap_switch==1:
			overlap_switch = 0
			
			A = np.asarray(block,dtype='f8')
			B = np.asarray(lastblock,dtype='f8')

			# Reshape A if necessary
			(m,n) = A.shape
			if m > dim:
				factor = m/dim
				dummy = A[0:dim,:]
				for k in range(factor-1):
					dummy = np.hstack([dummy,A[(k+1)*dim:(k+2)*dim]])
				A = dummy

			# Set the two pieces together
			if lastblock == []:
				overlap = A
			else: # standard case
				overlap = np.hstack([A,B])

			overlap_list.append(overlap)
			block = []
			lastblock = []

	infile.close()
	return overlap_list

def grep_core(filename):
	# Extracts the Core Hamiltonian from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	block = []
	lastblock = []
	core_list = []

	core_switch = 0
	for line in content:
		if "Core Hamiltonian" in line:
			core_switch = 1
		if core_switch==1 and '.' in line and "Final" not in line:
			dummy = line.split()
			if len(dummy) == 7:
				block.append(dummy[1:])
				dim = int(dummy[0]) # Will be correct when loop is through
			else:
				lastblock.append(dummy[1:])
		if "Final" in line and core_switch==1:
			core_switch = 0

			A = np.asarray(block,dtype='f8')
			B = np.asarray(lastblock,dtype='f8')

			# Reshape A if necessary
			(m,n) = A.shape
			if m > dim:
				factor = m/dim
				dummy = A[0:dim,:]
				for k in range(factor-1):
					dummy = np.hstack([dummy,A[(k+1)*dim:(k+2)*dim]])
				A = dummy

			# Set the two pieces together
			if lastblock == []:
				core = A
			else: # standard case
				core = np.hstack([A,B])

			core_list.append(core)
			block = []
			lastblock = []

	infile.close()

	return core_list
      
def grep_density(filename):
	# Extracts the density matrix from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	block = []
	lastblock = []
	density_list = []

	density_switch = 0
	for line in content:
		if " Final Alpha density matrix" in line:
			density_switch = 1
		if density_switch==1 and '.' in line and "Final" not in line:
			dummy = line.split()
			if len(dummy) == 7:
				block.append(dummy[1:])
				dim = int(dummy[0]) # Will be correct when loop is through
			else:
				lastblock.append(dummy[1:])
		if "Fock" in line and density_switch==1:
			density_switch = 0

			A = np.asarray(block,dtype='f8')
			B = np.asarray(lastblock,dtype='f8')

			# Reshape A if necessary
			(m,n) = A.shape
			if m > dim:
				factor = m/dim
				dummy = A[0:dim,:]
				for k in range(factor-1):
					dummy = np.hstack([dummy,A[(k+1)*dim:(k+2)*dim]])
				A = dummy

			# Set the two pieces together
			if lastblock == []:
				density = A
			else: # standard case
				density = np.hstack([A,B])

			density_list.append(density)
			block = []
			lastblock = []

	infile.close()

	return density_list

def grep_fock(filename):
	# Extracts the Fock Matrix from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	block = []
	lastblock = []
	fock_list = []

	fock_switch = 0
	for line in content:
		if "Final Alpha Fock Matrix" in line:
			fock_switch = 1
		if fock_switch==1 and '.' in line and "SCF" not in line:
			dummy = line.split()
			if len(dummy) == 5:
				block.append(dummy[1:])
				dim = int(dummy[0]) # Will be correct when loop is through
			else:
				lastblock.append(dummy[1:])
		if "SCF" in line and fock_switch==1:
			fock_switch = 0

			A = np.asarray(block,dtype='f8')
			B = np.asarray(lastblock,dtype='f8')

			# Reshape A if necessary
			(m,n) = A.shape
			if m > dim:
				factor = m/dim
				dummy = A[0:dim,:]
				for k in range(factor-1):
					dummy = np.hstack([dummy,A[(k+1)*dim:(k+2)*dim]])
				A = dummy

			# Set the two pieces together
			if lastblock == []:
				fock = A
			else: # standard case
				fock = np.hstack([A,B])

			fock_list.append(fock)
			block = []
			lastblock = []

	infile.close()
	return fock_list



def grep_MOs(filename):
	# Extracts MOs from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	block = []
	lastblock = []
	mos_list = []

	mo_switch = 0
	for line in content:
		sline = line[16:]
		if "MOLECULAR ORBITAL COEFFICIENTS" in line:
			mo_switch = 1
		if mo_switch==1 and '.' in line and "eigenvalues" not in line and "Mulliken Net Atomic" not in line:
			sdummy = sline.split()
			dummy = line.split()
			if len(sdummy) == 6:
				block.append(sdummy)
				dim = int(dummy[0]) # Will be correct when loop is through
			else:
				lastblock.append(sdummy)
		if "Mulliken Net Atomic" in line:
			mo_switch = 0
			A = np.asarray(block,dtype='f8')
			B = np.asarray(lastblock,dtype='f8')

			# Reshape A if necessary
			(m,n) = A.shape
			if m > dim:
				factor = m/dim
				dummy = A[0:dim,:]
				for k in range(factor-1):
					dummy = np.hstack([dummy,A[(k+1)*dim:(k+2)*dim]])
				A = dummy

			# Set the two pieces together
			if lastblock == []:
				mos = A
			else: # standard case
				mos = np.hstack([A,B])

			mos_list.append(mos)
			block = []
			lastblock = []

	infile.close()
	return mos_list
      
def grep_e_mo(filename):
	# Extracts MO-Energies from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	e_mos = []

	e_mos_switch = 0
	for line in content:
		if "-- Occupied --" in line:
			e_mos_switch = 1
			e_vec = []
		if e_mos_switch==1 and '.' in line and "-- Virtual --" not in line:
			dummy = line.split()
			for e in dummy:
			  e_vec.append(float(e))
		if "-- Virtual --" in line:
		 	e_mos_switch = 0
		 	e_mos.append(e_vec)

	infile.close()
	return e_mos

def grep_basis_info(filename):
	# Extracts the basis set information from a Q-Chem file

	infile = open(filename,"r")
	content = infile.readlines()

	atom_dict = {'S':1,'P':2,'D':3,'F':4,'G':5,'H':6}

	basis_switch = 0

	for line in content:
		if "[GTO]" in line:
			atom_indices = []
			basis_types = []
			basis_switch = 1
		if basis_switch==1:
			dummy = line.split()
			if len(dummy)==2 and '.' not in line:
				atom_index = dummy[0]
			if len(dummy)==3:
				basis_type = atom_dict[dummy[0]]
			if len(dummy)==2 and '.' in line:
				atom_indices.append(atom_index)
				basis_types.append(basis_type)
		if "[MO]" in line:
			basis_switch = 0

	infile.close()
	return (atom_indices,basis_types)
 
def grep_mull(filename):
     mull_switch = False
     mull = []
     with open(filename) as infile:
         for line in infile:
             if line.startswith('          Ground-State Mulliken Net Atomic Charges'):
                 mull_switch = True
                 #skip 3 lines
             if mull_switch and line.strip():
                 if '--' in line or 'Atom' in line:
                     if mull:
                         mull_switch = False
                     else:
                         continue
                 else:
                     sp = line.split()
                     mull.append(float(sp[2]))  
                     
     return mull            


def calc_energy(core,fock,mos):
	# Routine for calculating the SCF energy for a given set of MOs.
	# Note that Q-Chem adds 5 virtual MOs by default.
	# Note further, that the nuclear repulsion energy is not included. 

	(N_bas,N_mos) = np.shape(mos)

	E_elec = 0

	for k in range(N_mos-5):
		E_elec = E_elec + np.dot(mos[:,k].dot(fock+core),mos[:,k])

	return E_elec


def grep_aimd(inputfile):
	# Routine for the extraction of AIMD geometries, MOs and energies.
	# The geometries are converted to the zmat format using babel.

	a = qc.read(inputfile)
	
	# Get geometries in zmat format
	geo_list = []
	for geo in a.aimd.geometries:
		geo.write('dummy.xyz')
		os.system('babel dummy.xyz dummy.gzmat')

		infile = open('dummy.gzmat','r')
		content = infile.readlines()
		infile.close()
		os.system('rm dummy.xyz; rm dummy.gzmat')

		geometry = []
		for line in content:
			if '=' in line:
				dummy = line.split('=')
				geometry.append(float(dummy[1]))
		geometry = np.asarray(geometry)
		geo_list.append(geometry)
	
	# Get Overlap Matrices
	#overlap_list=grep_overlap(inputfile)
	
	# Get Core Hamiltonians
	core_list=grep_core(inputfile)

	# Get Fock Matrices
	fock_list=grep_fock(inputfile)
	
	# Get MO-Energies
	e_mos = grep_e_mo(inputfile)

	# Get MOs
	mos_list = grep_MOs(inputfile)

	# Get energies
	energies = a.aimd.energies

	# Get nuclear repulsion energies
	e_rep = grep_nuc_rep(inputfile)

    # Get basis set information
	(atom_indices,basis_types) = grep_basis_info(inputfile)

	# Q-Chem does extra SCF before AIMD starts,
	# so first entries of 'e_mos', 'mos', 'core', 'fock', 'overlap' and 'e_rep' have to be cut
	del e_rep[0]
	del e_mos[0]
	del mos_list[0]
	del fock_list[0]
	del core_list[0]
	#del overlap_list[0]

	return (geo_list,e_rep,core_list,fock_list,e_mos,mos_list,energies,atom_indices,basis_types)

def grep_sp(inputfile):
	# Routine for the extraction of AIMD geometries, MOs and energies.
	# The geometries are converted to the zmat format using babel.

	a = qc.read(inputfile, silent = True)
	
	# Get Overlap Matrices
	overlap=grep_overlap(inputfile)

	# Get Core Hamiltonians
	core=grep_core(inputfile)

	# Get Fock Matrices
	fock=grep_fock(inputfile)
	
	# Get MO-Energies
	e_mos = grep_e_mo(inputfile)

	# Get MOs
	#mos = grep_MOs_sp(inputfile)
	
	# Get Density Matrix
	density = grep_density(inputfile)

	# Get energies
	energy = a.general.energy

	# Get nuclear repulsion energies
	e_rep = grep_nuc_rep(inputfile)

    # Get basis set information
	#(atom_indices,basis_types) = grep_basis_info(inputfile)	

    # Get Mulliken charges
	mull = grep_mull(inputfile)

	return (e_rep,overlap,core,fock,e_mos,density,energy,mull)


def dens2molden(filename, geometry, e_mo, cis, style = 'molden'):
    basis = {'H' : ["S   3   1.00\n",
            "     3.42525091             0.15432897\n",     
            "     0.62391373             0.53532814\n",     
            "     0.16885540             0.44463454\n"] , 'C' :     
    ["S   3   1.00\n",
            "     71.6168370              0.15432897\n",       
            "     13.0450960              0.53532814\n",      
            "     3.5305122              0.44463454\n",     
            "SP   3   1.00\n",
            "     2.9412494             -0.09996723             0.15591627\n",      
            "     0.6834831              0.39951283             0.60768372\n",      
            "     0.2222899              0.70011547             0.39195739\n"]}  
    
    with open(filename,'w') as fout:
        fout.write('[Molden Format]\n')
        # Write atoms
        fout.write('[Atoms] Angs\n')
        if style == 'avogadro':
            fout.write('\n')
        for i, atom in enumerate(geometry.list_of_atoms):
            fout.write('{} {: 3} {: 3} {: 8} {: 8} {: 8}\n'.format(atom[0], i+1, 
                       qc.constants.dict_of_atomic_numbers[atom[0]],float(atom[1]),float(atom[2]),float(atom[3])))
            if style == 'avogadro':
                fout.write('\n')
        
        # Write basis
        fout.write('[GTO] \n')
        for i, atom in enumerate(geometry.list_of_atoms):
            fout.write('{} 0\n'.format(i+1))
            fout.writelines(basis[atom[0]])
            fout.write('\n')

        fout.write('[MO]\n')
        for e, ci in zip(e_mo,cis):
            fout.write('Sym= 1.1\nEne=   {}\nSpin= Alpha\nOccup=   2.000000\n'.format(e))
            for i, c in enumerate(ci):
                 fout.write('{: 3}  {}\n'.format(i+1, c))