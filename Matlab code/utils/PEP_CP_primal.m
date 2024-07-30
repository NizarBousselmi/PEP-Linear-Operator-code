%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 23-05-2023                                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input : - paramF : object with mu, L and type (e.g.                 %
%                    'SmoothStronglyConvex') of function f(x);        %
%         - paramG : object with mu, L and type of function g(y=Ax);  %
%         - paramA : object with mu, L and type ("sym", "skew",       %
%                    "nonsym") of operator A;                         %
%         - paramM : object with parameters of the method;            %
%                                                                     %
% output : - perf : performance of Chambolle-Pock method              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [perf,data] = PEP_CP_primal(paramF,paramG,paramA,paramM)

% Number of iterations
N = paramM.N;

% Initialize the PEP
P = pep();

% Initialize the functions f(x), g(Ax) and h(x)
F = P.DeclareFunction(paramF.type,paramF); 
G = P.DeclareFunction(paramG.type,paramG); 

% Starting points
x0 = P.StartingPoint(); % Primal starting point
u0 = P.StartingPoint(); % Dual starting point

% Optimality condition : A^T \nabla g(A*xs) + \nabla f(xs) = 0
xs = Point('Point');                % xs (primal optimal point)
ys = Point('Point');                % ys = A*xs
us = Point('Point');                % us = \nabla g(ys) (dual optimal point)
vs = Point('Point');                % vs = A^T \nabla g(ys)
ws = Point('Point');                % ws = \nabla f(xs)

fs = Point('Function value');       % fs = f(xs)
gs = Point('Function value');       % gs = g(ys)

F.AddComponent(xs,ws,fs); % f(xs) = fs and \nabla f(xs) = ws
G.AddComponent(ys,us,gs); % g(ys) = gs and \nabla g(ys) = us

P.AddConstraint((vs + ws)^2 == 0); % Optimality condition

% Declaration of x, y, u, v and Ffull
x = cell(N+2,1); % Iterates x 
y = cell(N+2,1); % Iterates y = Ax
u = cell(N+2,1); % Iterates u (gradient of g(y) for the gradient method and
                 % dual variables for primal-dual methods). u{N+2} is
                 % always the gradient of g at ys.
v = cell(N+2,1); % Iterates v = A^T u
%Ffull = cell(N+2,1); % Values of F(x) = f(x) + g(Ax)

% Initialization of x{i}, y{i}, u{i}, v{i} and Ffull{i} for i=1,N+2
x{1} = x0;               x{N+2} = xs;
y{1} = Point('Point');   y{N+2} = ys;
                         u{N+2} = us;
v{1} = Point('Point');   v{N+2} = vs;
%Ffull{1} = F.value(x{1}) + G.value(y{1}); Ffull{N+2} = fs + gs;

% Parameters and additional iterates of the method
u{1} = u0; %%%
%%%u{1} = G.gradient(y{1});%%%
sigma = paramM.sigma;
tau = paramM.tau;

if paramM.crit == "best"
    F_saved = cell(N,1);
end

% Apply the method
for i = 1:N
            
    y{i+1} = Point('Point');
    v{i+1} = Point('Point');
    
    % Chambolle-Pock method
    x{i+1} = proximal_step(x{i} - tau*v{i},F,tau);
    t = u{i} + sigma*(2*y{i+1}-y{i});
    u{i+1} = t - sigma*proximal_step(t/sigma,G,1/sigma);

    if paramM.crit == "best"
        F_saved{i}  = F.value(x{i+1}) +  G.value(y{i+1});
        P.PerformanceMetric(F_saved{i}-fs-gs,'min');
    end
    
end

% Performance criterion
if  paramM.crit == "mean"
    xsum = cell(1,1); xsum{1} = x{2};
    ysum = cell(1,1); ysum{1} = Point('Point');
    for i=2:N
        xsum{1} = xsum{1} + x{i+1};
    end
    xsum{1} = xsum{1}/(N);
   
    perf_crit = F.value(xsum{1}) - fs + G.value(ysum{1}) - gs;
    P.PerformanceMetric(perf_crit);
    
    points.x = [x;xsum];
    points.y = [y;ysum];
    points.u = u;
    points.v = v;
    
    %%%[utilde,~] = G.oracle(ysum{1});
    
elseif  paramM.crit == "half_mean"
    if mod(N,2) == 0 % N is even
        half_N = N/2;
    else
        half_N = (N+1)/2;
    end
    
    xsum = cell(1,1); xsum{1} = x{N+1};
    ysum = cell(1,1); ysum{1} = Point('Point');
    for i=1:(half_N-1)
        xsum{1} = xsum{1} + x{N+1-i};
    end
    xsum{1} = xsum{1}/(half_N);
   
    perf_crit = F.value(xsum{1}) - fs + G.value(ysum{1}) - gs;
    P.PerformanceMetric(perf_crit);
    
    points.x = [x;xsum];
    points.y = [y;ysum];
    points.u = u;
    points.v = v;
    
    %%%[utilde,~] = G.oracle(ysum{1});
    
elseif paramM.crit == "weighted_mean"
    xsum = cell(1,1); xsum{1} = 1*x{2};
    ysum = cell(1,1); ysum{1} = Point('Point');
    for i=2:N
        xsum{1} = xsum{1} + i*x{i+1};
    end
    xsum{1} = 2*xsum{1}/(N*(N+1));
   
    perf_crit = F.value(xsum{1}) - fs + G.value(ysum{1}) - gs;
    P.PerformanceMetric(perf_crit);
    
    points.x = [x;xsum];
    points.y = [y;ysum];
    points.u = u;
    points.v = v;
    
    %%%[utilde,~] = G.oracle(ysum{1});
        
elseif paramM.crit == "last"
 
    perf_crit = F.value(x{N+1}) - fs + G.value(y{N+1}) - gs;
    P.PerformanceMetric(perf_crit);
    
    points.x = x;
    points.y = y;
    points.u = u;
    points.v = v;

    %%%[utilde,~] = G.oracle(y{N+1});
    
elseif paramM.crit == "best"
   
    %perf_crit = F_saved{N}-fs-gs;
    %P.PerformanceMetric(F_saved{N}-fs-gs,'min');
    points.x = x;
    points.y = y;
    points.u = u;
    points.v = v;
    
    %%%[utilde,~] = G.oracle(y{N+1});
    
end

% Initial condition
P.InitialCondition( (x0-xs)^2 <= paramM.Rx^2);
P.InitialCondition( (us-u0)^2 <= paramM.Ru^2);%%%


% Link x,y,u,v through linear mapping
linear_operator_constraints(P,paramA,points);

% Solve the PEP
P.TraceHeuristic(0);
P.solve(0);
if paramM.crit == "best"
    value_solve = zeros(N,1);
    for i=1:N
        value_solve(i) = double(F_saved{i})-double(fs)-double(gs);
    end
    perf = min(value_solve);
else   
    perf = double(perf_crit);   % worst-case objective function accuracy
end


% Store the informations
if nargout > 1
    n = length(double(x{1}));
    m = length(double(y{1}));

    x_val = zeros(n,N+2); 
    y_val = zeros(m,N+2); 
    u_val = zeros(m,N+2);
    v_val = zeros(n,N+2);
    %f_val = zeros(1,N+2);

    for i=1:N+2
        x_val(:,i) = double(x{i}); % Iterates x of f(x)
        y_val(:,i) = double(y{i}); % Iterates y of g(y)
        u_val(:,i) = double(u{i}); % Iterates x of g(x)
        v_val(:,i) = double(v{i}); % Iterates x of f(x)
        %f_val(i) = double(f{i});
    end

    data.x = x_val ;
    data.y = y_val;
    data.u = u_val;
    data.v = v_val;
    %data.f = f_val;
end

end