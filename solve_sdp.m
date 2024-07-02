yalmip('clear')

q = 3;
N = 58;

m = N/2;
I = eye(m);
J = fliplr(I);
E = ones(m, m);
U = 1/sqrt(2) * [I I; J -J];
Z = zeros(m);

Rs = zeros(N, N, N);
Rs(:,:,1) = eye(N);
for i = 2:N
    Rs(:,:,i) = tril(circshift(Rs(:,:,i-1), -1, 2));
end

Rconjs = pagemtimes(pagemtimes(transpose(U), Rs), U);

tr = @(M, i) trace(Rs(:,:,i) * M);
tr_conj = @(M, i) trace(Rconjs(:,:,i) * M);

% Setting up the SDP
Cs = sdpvar(m, m, q+1); % C_i = (A_i + B_i * J) / 2
Ds = sdpvar(m, m, q+1); % D_i = (A_i - B_i * J) / 2

As = @(i) (Cs(:, :, i) + Ds(:, :, i)); % A_i = C_i + D_i 
Bs = @(i) (Cs(:, :, i) - Ds(:, :, i))*J; % B_i = (C_i - D_i) * J

Constraints = [
    Cs(:, :, 1) == 1/(2*N) * E * (I + J),...
    Ds(:, :, 1) == 1/(2*N) * E * (I - J),...
    Cs(:, :, q+1) == 1/(2*N) * I,...
    Ds(:, :, q+1) == 1/(2*N) * I,...
    % As(1) == E/N,... 
    % Bs(1) == E/N,...
    % As(q+1) == eye(m)/N,...
    % Bs(q+1) == zeros(m)
];

for t=1:q+1
    Constraints = [Constraints,...
        trace(As(t)) == 1/2,... % should try to vectorize
        Cs(:, :, t) >= 0,...
        Ds(:, :, t) >= 0
    ];
end


% TODO vectorize these as well
for t = 2:(q+1)
    Q_curr = blkdiag(Cs(:,:,t), Ds(:,:,t));
    Q_prev = blkdiag(Cs(:,:,t-1), Ds(:,:,t-1));
    % Q_curr = [As(t), Bs(t); J * Bs(t) * J, J * As(t) * J];
    % Q_prev = [As(t-1), Bs(t-1); J * Bs(t-1) * J, J * As(t-1) * J];

    for i=2:N 
        Constraints = [Constraints,...
            tr_conj(Q_curr - Q_prev, i) + (-1)^(t-1) * tr_conj(Q_curr - Q_prev, N + 2 -i) == 0
            % tr(Q_curr - Q_prev, i) + (-1)^(t-1) * tr(Q_curr - Q_prev, N + 2 -i) == 0
            ];
    end 
end 


Objective = 1;

options = sdpsettings('verbose', 1, 'solver', 'MOSEK');

optimize(Constraints, Objective, options)

% Avals = value(As);
% Bvals = value(Bs);
% 
% for t = 2:(q+1)
%     Q_curr = [Avals(:,:,t), Bvals(:,:,t); J * Bvals(:,:,t) * J, J * Avals(:,:,t) * J];
%     display(eig(Q_curr));
% end