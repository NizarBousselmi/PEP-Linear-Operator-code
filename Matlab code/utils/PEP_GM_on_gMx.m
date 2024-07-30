%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 23-05-2023                                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input : - paramG : object with mu, L and type (e.g.                 %
%                    'SmoothStronglyConvex') of function g(y);        %
%         - paramA : object with mu, L and type ("sym", "skew",       %
%                    "nonsym") of operator A;                         %
%         - paramM : object with number of iterations N and step size %
%                    h of gradient method;                            %
%         - R : initial distance ||x_0-x*||^2 <= R^2.                 %
%                                                                     %
% output : - perf : performance f(xN)-f* of (GM) on min g(Ax);        %
%          - data : contains the iterates, gradients and function     %
%                   values.                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [perf,data] = PEP_GM_on_gMx(paramG,paramA,paramM,R)

% Number of iterations
N = paramM.N;

% Initialize the PEP
P = pep();

% Initialize the function g(y) = g(Ax)
G = P.DeclareFunction(paramG.type,paramG); % g(y)

% Starting point
x0 = P.StartingPoint();

% Optimality condition : \nabla f(xs) = vs = 0
xs = Point('Point');                % xs
ys = Point('Point');                % ys = A*xs
us = Point('Point');                % us = \nabla g(ys)
vs = Point('Point',0);              % vs = A^T \nabla g(ys)
fs = Point('Function value');       % fs = g(ys) = F(xs)
G.AddComponent(ys,us,fs);

% Center the worst function
%P.AddConstraint(xs^2 == 0);
%P.AddConstraint(fs == 0);

% Initial condition
P.InitialCondition( (x0-xs)^2 <= R^2);

% Declaration of x, y, u, v and Ffull
x = cell(N+2,1); % Iterates x 
y = cell(N+2,1); % Iterates y = Ax
u = cell(N+2,1); % Iterates u (gradient of g at y)
v = cell(N+2,1); % Iterates v = A^T u
f = cell(N+2,1); % Values of F(x) = g(y)

% Initialization of x{i}, y{i}, u{i}, v{i} and f{i} for i=1,N+2
x{1} = x0;               x{N+2} = xs;
y{1} = Point('Point');   y{N+2} = ys;
u{1} = G.gradient(y{1}); u{N+2} = us;
v{1} = Point('Point');   v{N+2} = vs;
f{1} = G.value(y{1});     f{N+2} = fs;

% Parameters of the method
gamma = paramM.h/(paramA.L^2*paramG.L); % step size
% should be in linear_mapping.m but it works and requires less variables
theta = 1;
if paramA.type == "skew"
    theta = -1;
end

% Apply the algorithm
for i = 1:N
    
    x{i+1} = x{i} - theta*gamma*v{i};
    y{i+1} = Point('Point');
    u{i+1} = G.gradient(y{i+1});
    v{i+1} = Point('Point');
    
    f{i+1} = G.value(y{i+1});
    
end

% Performance criterion
fN = f{N+1};

% Links the iterates by a matrix
points.x = x;
points.y = y;
points.u = u;
points.v = v;

linear_operator_constraints(P,paramA,points);

% Solve the PEP
P.PerformanceMetric(fN-fs);      % Worst-case evaluated as g(yN)-g(ys)
P.TraceHeuristic(1);
P.solve(0);
perf = double(fN-fs);   % worst-case objective function accuracy

% Store the informations
if nargout > 1
    n = length(double(x{1}));

    x_val = zeros(n,N+2); 
    y_val = zeros(n,N+2); 
    u_val = zeros(n,N+2);
    v_val = zeros(n,N+2);
    f_val = zeros(1,N+2);

    for i=1:N+2
        x_val(:,i) = double(x{i}); % Iterates x of f(x)
        y_val(:,i) = double(y{i}); % Iterates y of g(y)
        u_val(:,i) = double(u{i}); % Iterates u
        v_val(:,i) = double(v{i}); % Iterates v
        f_val(i) = double(f{i});
    end

    data.x = x_val ;
    data.y = y_val;
    data.u = u_val;
    data.v = v_val;
    data.f = f_val;
end


end
