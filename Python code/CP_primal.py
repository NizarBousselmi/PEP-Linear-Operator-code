from PEPit import PEP
from PEPit import Point
from PEPit import Expression
from PEPit.functions import ConvexLipschitzFunction
from PEPit.operators import SymmetricLinearOperator
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step
import numpy as np


def wc_CP_primal(tau, sigma, L, Rx, Ru, n, crit, Mf, Mg, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x) + g(Mx),

    where :math:`f` and :math:`g` are convex and  :math:`\|M\|\leqL`.

    """

    # Instantiate PEP
    problem = PEP()

    # Declare two convex functions and a linear operator
    F = problem.declare_function(ConvexLipschitzFunction, M=Mf)
    G = problem.declare_function(ConvexLipschitzFunction, M=Mg) # g(y)
    M = problem.declare_function(function_class=LinearOperator, L=L)

    # Then define the starting point x0 and u0 of the algorithm
    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    
    # Defining unique optimal point xs = x_* of F(x) = g(Mx) and corresponding function value fs = f_*
    xs = Point();                # xs (primal optimal point)
    ys = M.gradient(xs);                # ys = A*xs
    us = Point();                # us = \nabla g(ys) (dual optimal point)
    vs = M.gradient_transpose(us)# vs = A^T \nabla g(ys)
    ws = Point();                # ws = \nabla f(xs)
    
    fs = Expression();       # fs = f(xs)
    gs = Expression();       # gs = g(ys)

    F.add_point((xs,ws,fs))  # f(xs) = fs and \nabla f(xs) = ws
    G.add_point((ys,us,gs))  # g(ys) = gs and \nabla g(ys) = us

    problem.add_constraint((vs + ws)**2 == 0.0); # Optimality condition
    
    # Run n steps of the CP method
    xi = x0
    ui = u0
    yi = M.gradient(xi)                   # y = Mx 
    
    xmoy = 0*x0
    ymoy = 0*yi
    
    xweight = 0*x0
    xmoy2 = 0*x0
    yweight = 0*yi
    ymoy2 = 0*yi

    
    for i in range(n):
        
        y_old = yi
        vi = M.gradient_transpose(ui) # v = M^T u
        xi, _, _ = proximal_step(xi-tau*vi, F, tau)
        yi = M.gradient(xi)               # y = Mx 
        t = ui + sigma*(2*yi-y_old)
        p, _, _ = proximal_step(t/sigma,G,1/sigma)
        ui = t - sigma*p
        
        if crit == "best":
            problem.set_performance_metric(F(xi) - fs + G(yi) - gs)
        else:
            xmoy = xmoy + xi
            ymoy = ymoy + yi
            
            xweight = xweight + float(i+1)*xi
            yweight = yweight + float(i+1)*yi
            
            if i+1 > np.floor(n/2.0):
                xmoy2 = xmoy2 + xi
                ymoy2 = ymoy2 + yi    
           
    if crit == "last":        
    # Last iteration
        xout = xi
        yout = yi
    elif crit == "mean":
    # Average iteration
        xout = xmoy/float(n)
        yout = ymoy/float(n)
    elif crit == "half_mean":
    # Average of half last iteration
        xout = xmoy2/float(np.ceil(0.5*n))
        yout = ymoy2/float(np.ceil(0.5*n))
    elif crit == "weighted_mean":
    # Weightened iteration
        xout = xweight/float(n*(n+1)*0.5)
        yout = yweight/float(n*(n+1)*0.5)
    
    if crit != "best": 
    # Set the performance metric to the primal value accuracy of xout
        problem.set_performance_metric(F(xout) - fs + G(yout) - gs)
    
    
    # Set the initial constraints 
    problem.set_initial_condition((x0 - xs)**2 <= Rx**2)
    problem.set_initial_condition((u0 - us)**2 <= Ru**2)
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Chambolle-Pock with fixed step-sizes ***')
        print('\tPEPit guarantee:\t {:.6}'.format(pepit_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau


if __name__ == "__main__":
    pepit_tau = wc_CP_primal(tau=1.0, sigma=1.0, L=1.0, Rx = 1.0, Ru = 1.0, n=10, crit="weighted_mean", Mf=1.0, Mg=1.0, verbose=1)
