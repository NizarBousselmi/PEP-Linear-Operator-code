%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 07-06-2024                                               %%%
%%% Note : Computes the worst-case primal dual gap of the average   %%%
%%%        primal-dual iterations of the Chambolle-Pock method on   %%%
%%%        convex functions with PEP. The initial bound is just     %%%
%%%        ||x-x0||^2 + ||u-u0||^2 <= R^2.                          %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils'))

%% PEP (w.r.t. N and tau=sigma)

n = 10; % Number of iterations
m = 5; % Number of different step size tested (tau = sigma)

tau_vec = linspace(0.2,1,m);
perf_PEP = zeros(n,m);

% Loop each step size tau_vec(k)
for k=1:m

    % Loop each number of iteration N
    for N=1:n 

    disp(N)
    
    tau = tau_vec(k);
    sigma = tau;
    LM = 1.;
    R = 1;

    % Initialize the PEP
    P = pep();

    % Initialize the functions f(x) and g(Mx)
    F = P.DeclareFunction('Convex'); 
    G = P.DeclareFunction('Convex'); 

    % Starting points
    x0 = P.StartingPoint(); % Primal starting point
    u0 = P.StartingPoint(); % Dual starting point

    % Declaration of xi, yi, ui, vi, x, y, u, v
    xi = cell(N+1,1); % Iterates xi 
    yi = cell(N+1,1); % Iterates yi = A xi
    ui = cell(N+1,1); % Iterates ui
    vi = cell(N+1,1); % Iterates vi = A^T ui

    x = cell(1,1); % Arbitrary x 
    y = cell(1,1); % Arbitrary y = A x
    u = cell(1,1); % Arbitrary u 
    v = cell(1,1); % Arbitrary v = A^T u

    % Initialization of xi{1}, yi{1}, ui{1}, vi{1}, x{1}, y{1}, u{1}, v{1}
    xi{1} = x0;  
    ui{1} = u0;
    yi{1} = Point('Point');                 
    vi{1} = Point('Point');

    x{1} = Point('Point');
    y{1} = Point('Point');
    u{1} = Point('Point');
    v{1} = Point('Point');

    % Apply the Chambolle-Pock algorithm
    for i = 1:N
        yi{i+1} = Point('Point');
        vi{i+1} = Point('Point');

        xi{i+1} = proximal_step(xi{i} - tau*vi{i},F,tau);
        t = ui{i} + sigma*(2*yi{i+1}-yi{i});
        ui{i+1} = t - sigma*proximal_step(t/sigma,G,1/sigma);

    end

    % Compute the average iterate
    xmoy = xi{2};
    umoy = ui{2};
    for i=1:(N-1)
        xmoy = xmoy + xi{i+2};
        umoy = umoy + ui{i+2};
    end
    xmoy = xmoy/N;
    umoy = umoy/N;

    % u0 \in \partial g(z)
    z = Point('Point');
    zmoy = Point('Point');

    G.AddComponent(z,u{1},Point('Function value')); 
    G.AddComponent(zmoy,umoy,Point('Function value'));

    points.x = [x;xi];
    points.y = [y;yi]; 
    points.u = [u;ui];
    points.v = [v;vi]; 

    % Primal-dual gap of the average iteration
    perf_crit = F.value(xmoy) - F.value(x{1}) + G.value(z) - G.value(zmoy) + v{1}*xmoy -u{1}*z -umoy*y{1} + umoy*zmoy;
    P.PerformanceMetric(perf_crit);

    % Initial condition 
    P.InitialCondition( (x0-x{1})^2 + (u{1}-u0)^2 <= R^2);

    % Link x,y,u,v through linear mapping
    paramM.type = "nonsym";
    paramM.L = LM;
    paramM.mu = 0;
    linear_operator_constraints(P,paramM,points);

    % Solve the PEP
    P.TraceHeuristic(0);
    P.solve(0);

    perf_PEP(N,k) = double(perf_crit);


    end
end

%% PEP the performance (w.r.t. N and tau=sigma)

clf

Nlin = linspace(1,N);

% Plot the performance computed by PEP for each step size
for k=1:m
    plot(1:N, perf_PEP(:,k),'.','Markersize',50); grid on; hold on
end

% Plot the identified curve 1/N+1
plot(Nlin,R^2./((Nlin+1)),'-','color','k','linewidth',1)

set(gca,'Fontsize',30)

ylabel('Primal-dual gap $\mathcal{L}(\bar{x}_N,u) - \mathcal{L}(x,\bar{u}_N)$','interpreter','latex')
xlabel('Number of iterations $N$','interpreter','latex')


legend({'$\tau = \sigma = 0.2$','$\tau = \sigma = 0.4$',...
       '$\tau = \sigma = 0.6$','$\tau = \sigma = 0.8$',...
       '$\tau = \sigma = 1$','$\frac{1}{N+1}R_0^2$'},'interpreter','latex')




