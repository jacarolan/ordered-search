yalmip('clear')

q = 3;
N = 200;


Rs = zeros(N, N, N);
Rs(:,:,1) = eye(N);
for i = 2:N
    Rs(:,:,i) = tril(circshift(Rs(:,:,i-1), -1, 2));
end

tr = @(M, i) trace(Rs(:,:,i) * M);

% Setting up the SDP
m = N/2;
diag_idx = (1:m+1:m^2).' + (0:q-1)*m^2;

I = eye(m);
J = fliplr(eye(m));
E = ones(m, m);

As = sdpvar(m, m, q+1); 
Bs = sdpvar(m, m, q+1, 'full');

function prod = pagemtimesLeft(M, Pgs)
    row_cnt = size(Pgs, 1);
    pg_cnt = size(Pgs, 3);
    prod = reshape(M * reshape(Pgs, row_cnt, []), row_cnt, row_cnt, pg_cnt);
end

function prod = pagemtimesRight(Pgs, M)
    prod = pagetranspose(pagemtimesLeft(transpose(M), pagetranspose(Pgs)));
end

Constraints = [
    As(:,:,1) == E/N,... 
    Bs(:,:,1) == E/N,...
    As(:,:,q+1) == eye(m)/N,...
    Bs(:,:,q+1) == zeros(m),...
    sum(As(diag_idx), 1) == 1/2,... 
    pagemtimesLeft(J, Bs) == pagetranspose(pagemtimesLeft(J, Bs)),...
    pagemtimesRight((As + Bs), J) >= 0,...
    pagemtimesRight((As - Bs), J) >= 0,...
];

% TODO vectorize these as well
for t = 2:(q+1)
    Q_curr = [As(:,:,t), Bs(:,:,t); J * Bs(:,:,t) * J, J * As(:,:,t) * J];
    Q_prev = [As(:,:,t-1), Bs(:,:,t-1); J * Bs(:,:,t-1) * J, J * As(:,:,t-1) * J];

    for i=2:N 
        Constraints = [Constraints,...
            tr(Q_curr - Q_prev, i) + (-1)^t * tr(Q_curr - Q_prev, N + 2 -i) == 0
            ];
    end 
end 


Objective = 1;

options = sdpsettings('verbose', 1, 'solver', 'MOSEK');

optimize(Constraints, Objective, options)