# distutils: language = c++
# distutils: sources = NonLinearModule/NonLinearModule_cpp.cpp

from eigency.core cimport *
# cimport eigency.conversions
# from NonLinearModule.eigency cimport *


# import eigency
# include "../eigency.pyx"

cdef extern from "NonLinearModule/NonLinearModule_cpp.h":

     cdef VectorXd _returnAlg "returnAlg"(Map[VectorXd] &, Map[MatrixXd] &, Map[MatrixXd] &, Map[MatrixXd] &, Map[VectorXd] &)


     cdef MatrixXd _Ktangent "Ktangent"(Map[VectorXd] &, Map[MatrixXd] &, Map[MatrixXd] &, Map[MatrixXd] &)



# Function with vector argument.
def returnAlg(np.ndarray[np.float64_t] array, np.ndarray[np.float64_t, ndim=2] array1, np.ndarray[np.float64_t, ndim=2] array2, np.ndarray[np.float64_t, ndim=2] array3, np.ndarray[np.float64_t] array4):
    return ndarray(_returnAlg(Map[VectorXd] (array), Map[MatrixXd] (array1), Map[MatrixXd] (array2), Map[MatrixXd] (array3), Map[VectorXd] (array4)))



# Function with vector argument.
def Ktangent(np.ndarray[np.float64_t] array, np.ndarray[np.float64_t, ndim=2] array1, np.ndarray[np.float64_t, ndim=2] array2, np.ndarray[np.float64_t, ndim=2] array3):
    return ndarray(_Ktangent(Map[VectorXd] (array), Map[MatrixXd] (array1), Map[MatrixXd] (array2), Map[MatrixXd] (array3)))




