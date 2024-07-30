function h0 = compute_h0(N,kappag)

M = 2*N+1;
c = 1-kappag;
xinit = 0.5;
fun = @(x)(1-M*x)*(1-x)^(-M) - c;
options = optimoptions('fsolve','Display','off');
x = fsolve(fun,xinit,options);
h0 = x/kappag;

assert(x<=1 && x>=0,'error in compute_h0');

end