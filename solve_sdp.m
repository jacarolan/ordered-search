yalmip('clear')

q = 3;
N = 100;


Rs = zeros(N, N, N);
Rs(:,:,1) = eye(N);
for i = 2:N
    Rs(:,:,i) = tril(circshift(Rs(:,:,i-1), -1, 2));
end

tr = @(M, i) trace(Rs(:,:,i) * M);

% Setting up the SDP
m = N/2;
I = eye(m);
J = fliplr(eye(m));
E = ones(m, m);

As = sdpvar(m, m, q+1); 
Bs = sdpvar(m, m, q+1, 'full');

Constraints = [
    As(:,:,1) == E/N,... 
    Bs(:,:,1) == E/N,...
    As(:,:,q+1) == eye(m)/N,...
    Bs(:,:,q+1) == zeros(m),...
    pagetraces(As) == 1/2,... 
    pagemtimes_left(J, Bs) == pagetranspose(pagemtimes_left(J, Bs)),...
    pagemtimes_right((As + Bs), J) >= 0,...
    pagemtimes_right((As - Bs), J) >= 0,...
];

% TODO vectorize these as well
for t = 2:(q+1)
    Q_curr = [As(:,:,t), Bs(:,:,t); J * Bs(:,:,t) * J, J * As(:,:,t) * J];
    Q_prev = [As(:,:,t-1), Bs(:,:,t-1); J * Bs(:,:,t-1) * J, J * As(:,:,t-1) * J];

    for i=2:N 
        Constraints = [Constraints,...
            tr(Q_curr - Q_prev, i) + (-1)^(t-1) * tr(Q_curr - Q_prev, N + 2 -i) == 0
            ];
    end 
end 


Objective = 1;

options = sdpsettings('verbose', 1, 'solver', 'MOSEK');

optimize(Constraints, Objective, options)