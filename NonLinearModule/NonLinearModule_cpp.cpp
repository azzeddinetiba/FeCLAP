#include "NonLinearModule_cpp.h"
#include <iostream>

using namespace Eigen;
yield::yield(VectorXd init_sigma, MatrixXd Tr, MatrixXd limit)
{
    //ctor
    double X, b, c;
    VectorXd a;
    RowVectorXd P1(3),P2(3);

    P1.setZero(3); P2.setZero(3);
    P1(0) = P1(1) = 1;
    a = limit*Tr*init_sigma;
    X = 1/limit(0,0);
    b = a.transpose()*a;
    c = (P1*Tr*init_sigma*P2*Tr*init_sigma);
    c *= (1/(X*X)) ;
    val = b-c-1.0;

}

void yield::update(VectorXd sigma, MatrixXd Tr, MatrixXd limit)
{
    //ctor
    double X, b;
    VectorXd a;
    RowVectorXd P1(3),P2(3);

    P1.setZero(3); P2.setZero(3);
    P1(0) = P1(1) = 1;
    a = limit*Tr*sigma;
    X = 1/limit(0,0);
    val = a.transpose()*a;
    b = (P1*Tr*sigma*P2*Tr*sigma);
    b *= (1/(X*X));
    val -= b + 1.0;


}

VectorXd yield :: grad (VectorXd sigma, MatrixXd Tr, MatrixXd limit)
{
    MatrixXd R2= MatrixXd::Zero(3,3),P3 = MatrixXd::Zero(3,3);
    VectorXd r;

    P3(0,1) = P3(1,0) = 1;
    R2(0,0) = 2*limit(0,0)*limit(0,0);
    R2(1,1) = 2*limit(1,1)*limit(1,1);
    R2(2,2) = 2*limit(2,2)*limit(2,2);

    r = Tr.transpose() * (R2 - limit(0,0)*limit(0,0)*P3)* Tr * sigma;

    return r;
}

MatrixXd yield :: grad2 (VectorXd sigma, MatrixXd TR, MatrixXd limit)
{
    MatrixXd r = MatrixXd::Zero(3,3);
    double X,Y,SLT;

    X=1/limit(0,0); Y=1/limit(1,1); SLT=1/limit(2,2);

    r(0,0) = (2*(SLT*SLT*X*X*TR(1, 0)*TR(1, 0)+SLT*SLT*Y*Y*TR(0, 0)*TR(0, 0)-SLT*SLT*Y*Y*TR(0, 0)*TR(1, 0)+X*X*Y*Y*TR(2, 0)*TR(2, 0)))/(X*X*Y*Y*SLT*SLT);
    r(0,1) = (2*SLT*SLT*X*X*TR(1, 0)*TR(1, 1)+2*SLT*SLT*Y*Y*TR(0, 0)*TR(0, 1)-SLT*SLT*Y*Y*TR(0, 0)*TR(1, 1)-SLT*SLT*Y*Y*TR(0, 1)*TR(1, 0)+2*X*X*Y*Y*TR(2, 0)*TR(2, 1))/(X*X*Y*Y*SLT*SLT);
    r(0,2) = (2*SLT*SLT*X*X*TR(1, 0)*TR(1, 2)+2*SLT*SLT*Y*Y*TR(0, 0)*TR(0, 2)-SLT*SLT*Y*Y*TR(0, 0)*TR(1, 2)-SLT*SLT*Y*Y*TR(0, 2)*TR(1, 0)+2*X*X*Y*Y*TR(2, 0)*TR(2, 2))/(X*X*Y*Y*SLT*SLT);

    r(1,0) = (2*SLT*SLT*X*X*TR(1, 0)*TR(1, 1)+2*SLT*SLT*Y*Y*TR(0, 0)*TR(0, 1)-SLT*SLT*Y*Y*TR(0, 0)*TR(1, 1)-SLT*SLT*Y*Y*TR(0, 1)*TR(1, 0)+2*X*X*Y*Y*TR(2, 0)*TR(2, 1))/(X*X*Y*Y*SLT*SLT);
    r(1,1) = (2*(SLT*SLT*X*X*TR(1, 1)*TR(1, 1)+SLT*SLT*Y*Y*TR(0, 1)*TR(0, 1)-SLT*SLT*Y*Y*TR(0, 1)*TR(1, 1)+X*X*Y*Y*TR(2, 1)*TR(2, 1)))/(X*X*Y*Y*SLT*SLT);
    r(1,2) = (2*SLT*SLT*X*X*TR(1, 1)*TR(1, 2)+2*SLT*SLT*Y*Y*TR(0, 1)*TR(0, 2)-SLT*SLT*Y*Y*TR(0, 1)*TR(1, 2)-SLT*SLT*Y*Y*TR(0, 2)*TR(1, 1)+2*X*X*Y*Y*TR(2, 1)*TR(2, 2))/(X*X*Y*Y*SLT*SLT);

    r(2,0) = (2*SLT*SLT*X*X*TR(1, 0)*TR(1, 2)+2*SLT*SLT*Y*Y*TR(0, 0)*TR(0, 2)-SLT*SLT*Y*Y*TR(0, 0)*TR(1, 2)-SLT*SLT*Y*Y*TR(0, 2)*TR(1, 0)+2*X*X*Y*Y*TR(2, 0)*TR(2, 2))/(X*X*Y*Y*SLT*SLT);
    r(2,1) = (2*SLT*SLT*X*X*TR(1, 1)*TR(1, 2)+2*SLT*SLT*Y*Y*TR(0, 1)*TR(0, 2)-SLT*SLT*Y*Y*TR(0, 1)*TR(1, 2)-SLT*SLT*Y*Y*TR(0, 2)*TR(1, 1)+2*X*X*Y*Y*TR(2, 1)*TR(2, 2))/(X*X*Y*Y*SLT*SLT);
    r(2,2) = (2*(SLT*SLT*X*X*TR(1, 2)*TR(1, 2)+SLT*SLT*Y*Y*TR(0, 2)*TR(0, 2)-SLT*SLT*Y*Y*TR(0, 2)*TR(1, 2)+X*X*Y*Y*TR(2, 2)*TR(2, 2)))/(X*X*Y*Y*SLT*SLT);

    return r;
}

yield::~yield()
{
    //dtor

}

double yield::getvalue()
{
    double r;
    r = val;
    return r;
}

VectorXd returnAlg (Map<VectorXd> &init_sigma, Map<MatrixXd> &Tr, Map<MatrixXd> &limit, Map<MatrixXd> &Q, Map<VectorXd> &delta_strain)

{
    returnIncr R;
    R.delta_lambda = 0; R.delta_sigma = VectorXd::Zero(3); R.sigma_state = VectorXd::Zero(3);
    double dlambda0, dlambda, q;
    VectorXd sol(4), b(3), dsigma0(3), dsigma(3), p(3), sigma_trial(3), state_sigma(3), f_deriv, e(3), ff(4), out(4);
    MatrixXd a(3,3), f_2deriv, a_tot(4,4);
    RowVectorXd c(3);

    sigma_trial = init_sigma + Q*delta_strain;
    yield f(sigma_trial, Tr, limit); q = f.getvalue();
    std::cout<<"first q is"<<std::endl;
    std::cout<<q;

    state_sigma = sigma_trial;


    if (q > 1e-7) //Plastic State
    {

        f_deriv = f.grad(sigma_trial, Tr, limit);
        dlambda0 = q/(f_deriv.transpose() * Q * f_deriv);
        dsigma0 = -dlambda0 * Q * f_deriv;

        // Initialization
        R.delta_lambda+= dlambda0;
        R.delta_sigma+= dsigma0;
        dsigma = dsigma0;

        for (int ii=0; ii<1000; ii++)
        {

            //Updating the yield function
            state_sigma = state_sigma + dsigma;
            f.update(state_sigma, Tr, limit);
            q = f.getvalue();


            //Checking the convergence criterion
            if (q < 1e-7)
            {
                break;
            }

            //Updating the yield limit derivatives
            //and the p parameter
            f_deriv = f.grad(state_sigma, Tr, limit);
            f_2deriv = f.grad2(state_sigma, Tr, limit);

            a = MatrixXd::Identity(3,3) + R.delta_lambda * Q * f_2deriv;
            b = Q * f_deriv;
            c = f_deriv.transpose();
            a_tot << a, b,
                     c, 0;
            p = R.delta_sigma - Q * delta_strain + R.delta_lambda * Q * f_deriv;
            ff << -p, -q;

            //Finding dsigma (i) and dlambda(i) for this increment
            sol =  a_tot.lu()  .solve(ff);
            dsigma  = sol.head(3);
            dlambda = sol(3);

            //Updating
            R.delta_lambda += dlambda;
            R.delta_sigma += dsigma;

            std::cout<<R.delta_lambda<<std::endl;
            std::cout<<R.delta_sigma<<std::endl;
            std::cout<<"q="<<std::endl;
            std::cout<<q<<std::endl;
            std::cout<<"then"<<std::endl;

            if (ii == 999 && q >1e-7)
            {
                throw std::overflow_error("The model could not converge");
            }

        }

    }

    //Saving the converged stress at yield = 0
    R.sigma_state = state_sigma;

    out << R.sigma_state, R.delta_lambda;
    return out;

}


MatrixXd Ktangent(Map<VectorXd> &result_sigma, Map<MatrixXd> &Tr, Map<MatrixXd> &limit, Map<MatrixXd> &Q)
{
    MatrixXd DcEp(3,3), Dc(3,3), f_2deriv(3,3);
    VectorXd f_deriv(3), state_sigma(3);
    double delta_lambda;

    state_sigma = result_sigma.head(3);
    delta_lambda = result_sigma(3);
    yield f(state_sigma, Tr, limit);
    f_deriv = f.grad(state_sigma, Tr, limit);
    f_2deriv = f.grad2(state_sigma, Tr, limit);

    Dc = Q.inverse() + delta_lambda * f_2deriv;
    Dc = Dc.inverse();

    DcEp = Dc * f_deriv * f_deriv.transpose() * Dc;
    DcEp = DcEp*(1/(f_deriv.transpose() * Dc * f_deriv));

    DcEp = Dc - DcEp;

    return DcEp;
}


