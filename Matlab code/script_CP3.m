%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 22-05-2023                                               %%%
%%% Note : This script computes the worst-case performance of the   %%%
%%%        Chambolle-Pock method applied to the problem min_x F(x)  %%%
%%%        where F(x) = f(x) + g(Ax), f and g are convex proximable %%%
%%%        with bounded subgradient and A has bounded norm          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils'))

%% Parameters

% Bounded subgradient on f(x)
paramF.type = 'ConvexBoundedGradient';
paramF.R = 1;

% Bounded subgradient on g(y)
paramG.type = 'ConvexBoundedGradient';
paramG.R = 1;

% Linear operator A 
paramA.L = 1;
paramA.mu = 0;
paramA.type = "nonsym"; % or "sym"

paramA.LA = paramA.L;
paramA.muA = paramA.mu;

% Parameters of Chambolle-Pock method
paramM.sigma = 1;
paramM.tau = 1;%1/(paramM.sigma*paramA.L^2);

% Bound on ||xs-x0|| and ||us-u0||
paramM.Rx = 1;
paramM.Ru = 1;

% PEP (w.r.t. N)
N = 10; % Number of iterations

perf_mean = zeros(N,1);
perf_last = zeros(N,1);
perf_best = zeros(N,1);
perf_half_mean = zeros(N,1);
perf_weighted_mean = zeros(N,1);

for i=1:N
    disp(i)
    paramM.N = i;
    
    paramM.crit = "mean";
    [perf_mean(i),data] = PEP_CP_primal(paramF,paramG,paramA,paramM);

    paramM.crit = "last";
    perf_last(i) = PEP_CP_primal(paramF,paramG,paramA,paramM);

    paramM.crit = "best";
    perf_best(i) = PEP_CP_primal(paramF,paramG,paramA,paramM);
    
    paramM.crit = "half_mean";
    perf_half_mean(i) = PEP_CP_primal(paramF,paramG,paramA,paramM);
    
    paramM.crit = "weighted_mean";
    perf_weighted_mean(i) = PEP_CP_primal(paramF,paramG,paramA,paramM);
    
end

%% Plot the performance (w.r.t. N)
clf

Nvec = 1:N;
Nlin = linspace(1,N);

s1 = semilogy(Nlin,5./Nlin,'-','color','k','linewidth',2); grid on; hold on;
s2 = semilogy(Nlin,1./sqrt(Nlin),'--','color','k','linewidth',2); 

s3 = semilogy(Nvec,perf_mean,'.','Markersize',40,'color','b'); grid on; hold on;
s4 = semilogy(Nvec,perf_last,'s','Markersize',18,'color','r','Linewidth',2);
s5 = semilogy(Nvec,perf_best,'.','Markersize',40,'color','g','Linewidth',3);
s6 = semilogy(Nvec,perf_half_mean,'.','Markersize',40,'color','m','Linewidth',3);
s7 = semilogy(Nvec,perf_weighted_mean,'.','Markersize',40,'color','k','Linewidth',3);

xlabel('Number of iterations $N$','interpreter','latex')
ylabel('Primal value accuracy','interpreter','latex')
legend([s3,s4,s5,s6,s7,s1,s2],{'$F(\bar{x}_N) - F^*$','$F(x_N)-F^*$',...
                                '$\min_{i\in I} F(x_i) - F^*$',...
                                '$F\left( \frac{1}{\left \lceil{N/2}\right \rceil } \sum_{i=\left \lfloor{N/2}\right \rfloor}^{N} x_i \right) - F^*$',...
                                '$F\left( \frac{2}{N(N+1)} \sum_{i=1}^{N} i x_i \right) - F^*$',...
                                '$\frac{5}{N}$',...
                                '$\frac{1}{\sqrt{N}}$'},'interpreter','latex')

set(gca,'Fontsize',30)

axis([1 N 0.085 4])

