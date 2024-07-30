from PEPit import PEP
from PEPit import Point
from PEPit import Expression
from PEPit.functions import ConvexFunction
from PEPit.operators import LinearOperator
from PEPit.primitive_steps import proximal_step
import numpy as np


def wc_CP(tau, sigma, L, R, n, init, verbose=1):
    """
    Consider the convex minimization problem

    .. math:: f_\\star \\triangleq \\min_x f(x) + g(Mx),

    where :math:`f` and :math:`g` are convex and  :math:`\|M\|\leqL`.

    """

    # Instantiate PEP
    problem = PEP()

    # Declare two convex functions and a linear operator
    F = problem.declare_function(ConvexFunction)
    G = problem.declare_function(ConvexFunction)
    M = problem.declare_function(function_class=LinearOperator, L=L)

    # Then define the starting point x0 and u0 of the algorithm
    x0 = problem.set_initial_point()
    u0 = problem.set_initial_point()
    
    x = Point()
    u = Point()
    
    # Run n steps of the CP method
    xi = x0
    ui = u0
    yi = M.gradient(xi)                   # y = Mx 
    
    xmoy = 0*x0
    umoy = 0*u0

    
    for _ in range(n):
        
        y_old = yi
        #vi = M.gradient_transpose(ui) # v = M^T u
        vi = M.gradient_transpose(ui) # v = M^T u
        xi, _, _ = proximal_step(xi-tau*vi, F, tau)
        yi = M.gradient(xi)               # y = Mx 
        t = ui + sigma*(2*yi-y_old)
        p, _, _ = proximal_step(t/sigma,G,1/sigma)
        ui = t - sigma*p
        
        xmoy = xmoy + xi
        umoy = umoy + ui
    
    # Compute the average
    xmoy = xmoy/float(n)
    umoy = umoy/float(n)
    
    # Define z : u0 \in \partial g(z)
    z = Point()
    zmoy = Point()
    G.add_point((z,u,Expression())) 
    G.add_point((zmoy,umoy,Expression())) 
    
    # Set the performance metric to the primal-dual gap of the average iteration
    problem.set_performance_metric(F(xmoy) - F(x) + G(z) - G(zmoy) + M.gradient_transpose(u)*xmoy - u*z - umoy*M.gradient(x) + umoy*zmoy)
    
    # Set the initial constraint 
    if init == "classic":
        problem.set_initial_condition((1.0/tau)*(x0 - x)**2 + (1.0/sigma)*(u0-u)**2 - 2*(x-x0)*(M.gradient_transpose(u)-M.gradient_transpose(u0)) <= R**2)
    elif init == "reduced":
        problem.set_initial_condition((x0 - x)**2 + (u0-u)**2 <= R**2)
    
    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Compute theoretical guarantee
    theoretical_tau = R**2/(2*(n+1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of Chambolle-Pock with fixed step-sizes ***')
        print('\tPEPit guarantee:\t {:.6}'.format(pepit_tau))
        print('\tTheoretical guarantee:\t {:.6} '.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_CP(tau=0.4, sigma=0.4, L=1.0, R=1.0, n=5, init = "reduced", verbose=1)
