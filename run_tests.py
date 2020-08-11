from numpy.testing import assert_array_equal
import numpy as np
import NonLinearModule

def show_the_matrix():
        x = 1e9*np.array([0,0, 0])
        Tr= np.eye(3)
        limit=(1/(50e6))*Tr
        Q = 2e11 * np.eye(3)
        delta_strain = np.array([0.001, 0., 0.])


        cpp_size = NonLinearModule.returnAlg(x, Tr, limit, Q, delta_strain)
        ha = cpp_size.reshape((1, 4))
        ha =np.array(ha[0])
        new_Q = NonLinearModule.Ktangent ( ha,  Tr,  limit, Q)
        print(new_Q)
        # Shared memory test: Verify that first entry was set to 0 by C++ code.


if __name__ == '__main__':
    show_the_matrix()
