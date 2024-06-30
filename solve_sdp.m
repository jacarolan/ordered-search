yalmip('clear')

q = 6;
N = 100;

Rs = zeros(N, N, N+1);
Rs(:,:,1) = eye(N);
for i = 1:N
    Rs(:,:,i+1) = tril(circshift(Rs(:,:,i), -1, 2));
end

% Setting up the SDP
I = eye(N/2);
J = fliplr(eye(N/2));
E = ones(N/2, N/2);

As = sdpvar(N/2, N/2, q+1); 
Bs = sdpvar(N/2, N/2, q+1, 'full');

Constraints = [
    As(:,:,1) == E/N,... 
    Bs(:,:,1) == E/N,...
    As(:,:,q) == eye(N/2)/N,...
    Bs(:,:,q) == zeros(N/2)
];


m = N/2;
diag_idx = (1:m+1:m^2).' + (0:q-1)*m^2;
Constraints = [Constraints, sum(As(diag_idx), 1) == 1/2];

% TODO vectorize all of these 
for i = 1:q
    Constraints = [Constraints, J * Bs(:,:,i) * J == transpose(Bs(:,:,i))];

    Constraints = [Constraints, (As(:,:,i) + Bs(:,:,i)) * J >= 0];
    Constraints = [Constraints, (As(:,:,i) - Bs(:,:,i)) * J >= 0];
end

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