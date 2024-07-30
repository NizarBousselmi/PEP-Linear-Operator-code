%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Authors: Nizar Bousselmi, Julien Hendrickx and Francois Glineur %%%
%%% Date : 23-05-2023                                               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input : - P : instance of pep();                                    %
%         - paramA : object with mu, L and type ("sym", "skew",       %
%                    "nonsym") of operator A;                         %
%         - points : object with the set of points x, y, u and v;     %
%                                                                     %
% output : no output, it adds the LMI constraint to PEP, guaranteeing %
%          that y = Ax and v = A^T u with mu <= ||A|| <= L.           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function linear_operator_constraints(P,paramA,points)

x = points.x;
y = points.y;
u = points.u;
v = points.v;
N1 = length(x);
N2 = length(u);

assert(N1 == length(y));
assert(N2 == length(v));

L = paramA.L;
mu = paramA.mu;

% Symmetric or skew-symmetric
if paramA.type ==  "sym" || paramA.type == "skew"
    
%    assert(N1==N2); not necessary
    x2 = [x;u];
    y2 = [y;v];
    
    theta = 1;
    if paramA.type == "skew"
        % Requiring singular values between 0,L means eigenvalues between
        % -iL and iL (we dont need to put mu = -iL because X^T Y = Y^T X)
        mu = 1i*L;
        L = 1i*L;
        theta = -1;
    end
    
    M = cell(N1+N2);
    for i=1:(N1+N2)
        for j=1:(N1+N2)
            if j ~= i
                % B = B^T or B = -B^T
                P.AddConstraint(x2{i}*y2{j} == theta*y2{i}*x2{j});
            end
                % Build M = -muL A + (mu+L)B - C
                M{i,j} = (y2{i} - mu*x2{i})*(L*x2{j} - y2{j});
        end
    end

    P.AddLMIConstraint(M); % Add an LMI constraint for M
    
elseif paramA.type == "nonsym"
    % B1 = B2
    for i=1:N1
        for j=1:N2
            P.AddConstraint(x{i}*v{j} == u{j}*y{i});
        end
    end

    tol = 0;
    % C1 <= S^2 C2
    Ma = cell(N2);
    for j1=1:N2
        for j2=1:N2
            if j2==j1
                Ma{j1,j2} = L*L*u{j1}*u{j2}-v{j1}*v{j2}-tol;
            else
                Ma{j1,j2} = L*L*u{j1}*u{j2}-v{j1}*v{j2};
            end
        end
    end
    P.AddLMIConstraint(Ma);

    % A2 <= S^2 A1
    Mb = cell(N1);
    for i1=1:N1
        for i2=1:N1
            if i1 == i2
                Mb{i1,i2} = L*L*x{i1}*x{i2}-y{i1}*y{i2}-tol;
            else
                Mb{i1,i2} = L*L*x{i1}*x{i2}-y{i1}*y{i2};
            end
        end
    end
    P.AddLMIConstraint(Mb); % Add an LMI constraint for M

else
    error('undefined type of symmetry')
end

end