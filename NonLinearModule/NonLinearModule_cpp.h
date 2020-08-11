#include <Eigen/Dense>
#include <Eigen/LU>
#ifndef YIELD_H
#define YIELD_H

using namespace Eigen;

/* A class to get the value of the yield function,
its derivative and its second derivative */
class yield
{
    public:
        yield(VectorXd , MatrixXd, MatrixXd);
        ~yield();
        VectorXd grad (VectorXd , MatrixXd, MatrixXd);
        MatrixXd grad2 (VectorXd , MatrixXd, MatrixXd);
        void update (VectorXd, MatrixXd, MatrixXd);
        double getvalue();
    private:

        double val;
};


/* A structure-type data that contains the
plastic consistency parameter and the stress increment */
struct returnIncr
{
    double delta_lambda;
    VectorXd delta_sigma;
    VectorXd sigma_state;
};


VectorXd returnAlg (Map<VectorXd> &init_sigma, Map<MatrixXd> &Tr, Map<MatrixXd> &limit, Map<MatrixXd> &Q, Map<VectorXd> &delta_strain);

MatrixXd Ktangent(Map<VectorXd> &result_sigma, Map<MatrixXd> &Tr, Map<MatrixXd> &limit, Map<MatrixXd> &Q);



#endif // YIELD_H
