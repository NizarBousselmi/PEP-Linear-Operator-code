%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 22-05-2023                                               %%%
%%% Note : This script computes the worst-case performance of the   %%%
%%%        gradient method applied to the class of functions of the %%%
%%%        form F(x) = g(Ax) where g(y) is Lg-smooth mug-strongly   %%%
%%%        convex and A is a linear operator with muA<=||A||<=LA,   %%%
%%%        through Performance Estimation Problem.                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils'))

%% PEP (w.r.t. h)
% Compute the worst-case performance of the gradient method applied
% F(x) = g(Ax) for different step sizes by Performance Estimation Problem

% Parameter of function g(y) and operator A
paramG.mu = 0.1;
paramG.L = 1;
paramG.type = 'SmoothStronglyConvex';
paramA.L = 1;
paramA.mu = 0;
paramA.type = "sym"; % or "nonsym"

paramM.N = 10; % Number of iterations
R = 1;         % Initial distance
n = 1;         % Number of step size analyzed by PEP
hvec = linspace(0.,2,n);

% Computation of the worst-case by PEP
perf_PEP = zeros(n,1);
for i=1:n
    disp(i)
    paramM.h = hvec(i);
    [perf_PEP(i),~] = PEP_GM_on_gMx(paramG,paramA,paramM,R);
end

%% Plot the performance (w.r.t. h)
clf

LA = paramA.L;     muA = paramA.mu;      kappaA = muA/LA;
L = LA^2*paramG.L; mu = paramG.mu*muA^2; kappa = mu/L; mug = paramG.mu;
kappag = paramG.mu/paramG.L;
N = paramM.N;
hlin = linspace(0,2,1000);

% Performance on F_mug,Lg,muA,LA by PEP
%semilogy(hvec,perf_PEP,'.','color','b','Markersize',50); hold on; grid on;

% Performance on F_0,L
p2 = semilogy(hlin,R*R*0.5*L*max([1./(2*N*hlin+1); ((1-hlin).^2).^N]),':','Linewidth',4,'color','r'); hold on; grid on;

% Performance on F_mug,Lg
p3 = semilogy(hlin,R*R*max([0.5*L*kappag./(kappag-1+(1-kappag*hlin).^(-2*N)); 0.5*L*(1-hlin).^(2*N)]),'-.','Linewidth',4,'color','k');

% Analytical expression of the performance
h0 = compute_h0(N,kappag);
M = sqrt(h0./hlin); M(M<kappaA) = kappaA; M(M>1) = 1;
p1 = plot(hlin,R*R*0.5*L*max([M.^2.*kappag./(kappag-1+(1-M.^2.*kappag.*hlin).^(-2*N)); (1-hlin).^(2*N)]),'Linewidth',3,'color','b');

legend([p2 p1 p3],{"$w(\mathcal{F}_{0};h)$",...
                   "$w(\mathcal{C}_{"+mug+"}^{"+muA+"};h )$",...
                   "$w(\mathcal{F}_{"+mug+"};h)$"},'interpreter','latex','Fontsize',30)

set(gca,'Fontsize',40)
xlabel('Step size $h$','interpreter','latex','Fontsize',50)
ylabel("Accuracy $F(x_{"+N+"})-F^*$",'interpreter','latex','Fontsize',50)

