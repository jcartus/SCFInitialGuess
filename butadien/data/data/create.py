# pyQChem script for a AIMD follow-up with reduced basis set size
#
# AWH/RM 2015

import pyQChem as qc

# First, read all N geometries from aimd outputfile 
a = qc.read('aimd_small.out')
aa = a.list_of_jobs[1]

# Second, create the template 
myrem = qc.rem_array()
myrem.jobtype('sp')
myrem.exchange('hf')
myrem.basis('sto-3g')
myrem.scf_algorithm('rca_diis')
#myrem.max_rca_cycles('10')
#myrem.thresh_rca_switch('7')
myrem.scf_guess('sad')
myrem.scf_convergence('9')
myrem.thresh('14')
myrem.incfock('0')
myrem.incdft('0')
#myrem.max_scf_cycles('500')
#myrem.symmetry('false')
#myrem.sym_ignore('true')
myrem.mem_total('16000')
myrem.mem_static('4000')

# Add rem_array
myfile = qc.inputfile()
myfile.add(myrem)

# Add geometry array
mygeo  = qc.mol_array(aa.aimd.geometries[0])
myfile.add(mygeo)

# Finally, create each inputfile and write it to disk
for i,k in enumerate(aa.aimd.geometries):
	mygeo  = qc.mol_array(k)
	myfile.add(mygeo)
	filename = str(i) + ".inp"
	myfile.write(filename)






