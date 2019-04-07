"""This file contains descriptor classes that use the direction of one 
atom to another in the molecule to descripte the environment

Author:
    Johannes Cartus, QCIEP, TU Graz
"""

import numpy as np
from scipy.special import sph_harm




#-------------------------------------------------------------------------------
#   Low level descriptor classes
#-------------------------------------------------------------------------------

class EmptyGaussians(object):

    def __init__(self):
        
        self.number_of_descriptors = 1
        

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vector will be list!!!!
        """
        return [ 1 ]        


    def calculate_inverse_descriptor(self, t, y):
        """Returns the y- weighted sum of the gaussians,
        evaluated at values t.
        """
        return 1

class Gaussians(object):

    def __init__(self, r_s, eta):
        
        # check if list are of equal length of if eta is scalar
        # check if list are of equal length of if eta is scalar
        if isinstance(eta, (list, tuple)):
            if len(r_s) != len(eta) and len(eta) != 1 :
                raise ValueError("Dimension of r_s and eta do not match")

        self.r_s = np.array(r_s)
        self.eta = np.array(eta)

        self.number_of_descriptors = len(r_s)

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vector will be list!!!!
        """
        return [
            np.exp(-1 * eta*(x - r_s)**2) \
                for (eta, r_s) in zip(self.eta, self.r_s)
        ]        


    def calculate_inverse_descriptor(self, t, y):
        """Returns the y- weighted sum of the gaussians,
        evaluated at values t.
        """
        return np.dot(
            np.array(y).reshape(-1), 
            np.array(self.calculate_descriptor(t))
        )


def periodic_gaussian(x, r_s, eta, period):
    """Gaussian that with width eta, centered at r_s, periodic with period 
    period, evaluated at x"""
    return np.maximum(
        np.exp(-1 * eta * ((x - r_s) % period)**2),
        np.exp(-1 * eta * (((x-r_s) % period) - period)**2) 
    )
            

class PeriodicGaussians(Gaussians):

    def __init__(self, r_s, eta, period):
        """Constructor
        Args:
            - r_s list<float>: centers of gaussians.
            - eta list<float> or float: width of gaussians.
            - period float: the period after which the 
                gaussians are repeated.
        """
        
        # check if list are of equal length of if eta is scalar
        if isinstance(eta, (list, tuple)):
            if len(r_s) != len(eta) and len(eta) != 1 :
                raise ValueError("Dimension of r_s and eta do not match")

        self.r_s = np.array(r_s)
        self.eta = np.array(eta)
        self.period = period
        
        self.number_of_descriptors = len(r_s)

    def calculate_descriptor(self, x):
        """Returns a descriptor value for an abstract quantity x (e.g. a 
        distance or an angle). The vector will be list!!!!
        """
        return [
            periodic_gaussian(x, r_s, eta, self.period) \
            for (r_s, eta) in zip(self.r_s, self.eta)
        ]
    
class ConstantAngularDescriptor(object):
    """Descriptor for the angular part. Returns no descriptors. Inverse
    descriptor is always just 1.
    """

    def __init__(self):
        """Constructur.
        """


        self.number_of_descriptors = 0

    def calculate_descriptor(self, *args):
        """Calculates angular descriptor part. No values are added 
        to the symmetry vector"""

        #Azimuthal and polar descriptors must return a list.
        return  []

    def calculate_inverse_descriptor(self, *args):

        return 1


class IndependentAngularDescriptor(object):
    """Descriptor for the angular part. Here azimuthal and polar 
    angle are described independently.
    """

    def __init__(self, azimuthal_descriptor, polar_descriptor):
        """Constructur.

        Args:
            - azimuthal_descriptor <AbstractQuantityDescriptor>: can e.g. a 
            periodic Gaussian.
            - polar_descriptor <AbstractQuantityDescriptor>: can e.g. a 
            periodic Gaussian.
        """

        self.azimuthal_descriptor = azimuthal_descriptor
        self.polar_descriptor = polar_descriptor

    @property 
    def number_of_descriptors(self):
        return self.azimuthal_descriptor.number_of_descriptors + \
            self.polar_descriptor.number_of_descriptors

    def calculate_descriptor(self, r, phi, theta):
        """Calculates angular descriptor part. """

        #Azimuthal and polar descriptors must return a list.
        return  list(self.azimuthal_descriptor.calculate_descriptor(phi)) + \
            list(self.polar_descriptor.calculate_descriptor(theta))

    def calculate_inverse_descriptor(self, r, phi, theta, y):

        return self.azimuthal_descriptor.calculate_inverse_descriptor(
            phi, 
            y[:self.azimuthal_descriptor.number_of_descriptors]
        ) * self.polar_descriptor.calculate_inverse_descriptor(
            theta, 
            y[self.azimuthal_descriptor.number_of_descriptors:]
        )


class SPHAngularDescriptor(object):
    """
    Real and Imaginary part of the spherical harmonics 
    """

    def __init__(self, l_max):
        
        
        self.l_max = l_max

    @property
    def number_of_descriptors(self):
        return int((self.l_max + 1)**2 * 2)


    def calculate_descriptor(self, r, phi, theta):
        """Returns a vector with the spherical harmonics of 
        the direction specified by phi an theta. The result is a list
        of first the real and then the imaginary part.
        """


        real, imaginary = [], []
        for l in range(self.l_max + 1):
            for m in range(-l, l + 1):
                sph = sph_harm(m, l, phi, theta)
                real.append(sph.real)
                imaginary.append(sph.imag)
        
        return np.array(real + imaginary)

    def calculate_inverse_descriptor(self, r, phi, theta, y):
        """Y is actually G (symmetry vector)."""

        sph = self.calculate_descriptor(r, phi, theta) 

        n_functions = self.number_of_descriptors // 2

        real = np.dot(y[:n_functions].T, sph[:n_functions])
        imag = np.dot(y[n_functions:].T, sph[n_functions:])

        tmp = np.sqrt(real**2 + imag **2)

        return np.sqrt(real**2 + imag **2).reshape(-1)



#todo better name
class SphereSectionDescriptor(object):
    """Angles of space are divided into sections, defining the bounds of bins.
    Every bin has its own angular description. Should be used together with 
    a mock for the radial descriptor because radial description is contained 
    in this descriptor.
    """

    def __init__(self, 
        number_polar_sections, 
        number_azimuthal_sections,
        radial_descriptor
    ):

        #--- polar ---
        self.number_polar_sections = number_polar_sections
        #---

        #--- azimuthal ---
        self.number_azimuthal_sections = number_azimuthal_sections
        #--- 

        # radial
        self.radial_descriptor = radial_descriptor

        # overall
        self.number_of_descriptors = \
            self.number_polar_sections * \
                self.number_azimuthal_sections * \
                    self.radial_descriptor.number_of_descriptors
    
    @staticmethod
    def _calculate_section(x, number_of_sections, period_x):
        """Calculates section for an abstract value x, periodic with period_x,
        which is sectioned in parts of dx.

        The first section is centered at 0, i.e. it goes from [-dx, dx).
        """
        
        dx = period_x / number_of_sections

        return int(np.floor(((x+0.5*dx) % period_x) / dx))
    
    def _calculate_index_polar(self, theta):
        """Calculates which polar section (from 0 to number of polar sections-1)
        the angle theta would fall into."""
        return self._calculate_section(
            theta, 
            self.number_polar_sections, 
            np.pi
        )

    def _calculate_index_azimuthal(self, phi):
        """Calculates which azimuthal section (from 0 to number of azimuthal 
        sections-1) the angle phi would fall into."""
        return self._calculate_section(
            phi,
            self.number_azimuthal_sections,
            2 * np.pi
        )
    
    def _calculate_symmetry_vector_range(self, index_polar, index_azimuthal):
        """Calculate the range (start, end) of elements in the symmetry 
        vector, that correspond to the radial description of the 
        section that corresponds to the given indices for polar/azimuthal."""

        start = (index_polar * self.number_azimuthal_sections \
            + index_azimuthal) * self.radial_descriptor.number_of_descriptors
        
        end = start + self.radial_descriptor.number_of_descriptors

        return start, end

    def calculate_descriptor(self, r, phi, theta):
        """Returns a vector with the spherical harmonics of 
        the direction specified by phi an theta. The result is a list
        of first the real and then the imaginary part.
        """

        g = np.zeros(self.number_of_descriptors)

        #--- find which section this atom will contribute to ---
        index_polar = self._calculate_index_polar(theta)
        index_azimuthal = self._calculate_index_azimuthal(phi)
        #---

        #--- calculate where this has to be placed in result vector ---
        start, end = \
            self._calculate_symmetry_vector_range(index_polar, index_azimuthal)
        #---

        g[start:end] = self.radial_descriptor.calculate_descriptor(r)

        return g

    
    def calculate_inverse_descriptor(self, r, phi, theta, y):

        try: 
            test = r[0]
        except:
            r = [r]

        #--- calculate radial activation for all sections ---
        #
        # result: activation_by_section.shape = \
        #  (len(number_polar_sections), len(number_azimuthal_sections), len(r))

        activation_by_section = []

        for i in range(self.number_polar_sections):
            tmp = []
            for j in range(self.number_azimuthal_sections):

                start, end = self._calculate_symmetry_vector_range(i, j)

                tmp.append(
                    self.radial_descriptor.calculate_inverse_descriptor(
                        r, 
                        y[start:end]
                    )
                )

            activation_by_section.append(tmp)
        #---


        #--- map the radial activation to new array ---
        # (according to the section phi and theta belong to)
        #
        # result: activation.shape = (len(theta), len(phi), len(r))
        try:
            if len(phi) > 1:
                pass
        except:
            phi = [phi]
        
        try: 
            if len(theta) > 1:
                pass
        except:
            theta = [theta]


        activation = []

        for i in range(len(theta)):
            
            index_polar = self._calculate_index_polar(theta[i])
            tmp = []

            for j in range(len(phi)):
                index_azimuthal = self._calculate_index_azimuthal(phi[j])

                tmp.append(
                    activation_by_section[index_polar][index_azimuthal]
                )

            activation.append(tmp)
        #---

        return np.array(activation)





