yalmip('clear')

q = 6;
N = 100;

Rs = zeros(N, N, N+1);
Rs(:,:,1) = eye(N);
for i = 1:N
    Rs(:,:,i+1) = tril(circshift(Rs(:,:,i), -1, 2));
end

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
    As(:,:,q) == eye(m)/N,...
    Bs(:,:,q) == zeros(m),...
    sum(As(diag_idx), 1) == 1/2,... 
    pagemtimesLeft(J, Bs) == pagetranspose(pagemtimesLeft(J, Bs)),...
    pagemtimesRight((As + Bs), J) >= 0,...
    pagemtimesRight((As - Bs), J) >= 0,...
];

% TODO vectorize these as well
for t = 2:q
    Q_curr = [As(:,:,t), Bs(:,:,t); J * Bs(:,:,t) * J, J * As(:,:,t) * J];
    Q_prev = [As(:,:,t-1), Bs(:,:,t-1); J * Bs(:,:,t-1) * J, J * As(:,:,t-1) * J];

    for i=1:N 
        Constraints = [Constraints, (trace((Rs(:,:,i+1) + (-1)^t * Rs(:,:,N-i+1)) * (Q_curr - Q_prev)) == 0)];
    end 
end 


Objective = 0;

options = sdpsettings('verbose', 1, 'solver', 'SCS');

sol = optimize(Constraints, Objective, options);