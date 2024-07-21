yalmip('clear')

USE_NEW_CONSTRAINTS = false;

q = 5;
N = 200;

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
Obj_to_maximize = 0;
Cs = sdpvar(m, m, q+1); % C_i = (A_i + B_i * J) / 2
Ds = sdpvar(m, m, q+1); % D_i = (A_i - B_i * J) / 2

As = @(i) (Cs(:, :, i) + Ds(:, :, i)); % A_i = C_i + D_i 
Bs = @(i) (Cs(:, :, i) - Ds(:, :, i))*J; % B_i = (C_i - D_i) * J
Qs = @(i) [As(i), Bs(i); J * Bs(i) * J, J * As(i) * J]; % B_i = (C_i - D_i) * J

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
    Obj_to_maximize = Obj_to_maximize + Cs(1,1,t) + Ds(1,1,t);
    Constraints = [Constraints,...
        trace(As(t)) == 1/2,... % should try to vectorize
        Cs(:, :, t) >= 0,...
        Ds(:, :, t) >= 0
    ];
end


% TODO vectorize these as well
for t = 2:(q+1)
    step_is_odd  = (rem(t, 2) == 0); % indexing starts at 1... 
    step_is_last = (t == q+1);

    Q_curr = blkdiag(Cs(:,:,t), Ds(:,:,t));
    Q_prev = blkdiag(Cs(:,:,t-1), Ds(:,:,t-1));
    % Q_curr = [As(t), Bs(t); J * Bs(t) * J, J * As(t) * J];
    % Q_prev = [As(t-1), Bs(t-1); J * Bs(t-1) * J, J * As(t-1) * J];

    if USE_NEW_CONSTRAINTS && step_is_odd && ~step_is_last
        Q_next = blkdiag(Cs(:, :, t+1), Ds(:, :, t+1));
        for i=2:N
            Constraints = [Constraints,...
                tr_conj(2*Q_curr - Q_prev - Q_next, i) == -tr_conj(Q_prev - Q_next, N + 2 - i)
            ];
        end
    end

    if ~USE_NEW_CONSTRAINTS || (step_is_odd && step_is_last)
        for i=2:N 
            Constraints = [Constraints,...
                tr_conj(Q_curr - Q_prev, i) + (-1)^(t-1) * tr_conj(Q_curr - Q_prev, N + 2 -i) == 0
                ];
        end
    end
end 

options = sdpsettings('verbose', 1, 'solver', 'MOSEK');

optimize(Constraints, -Obj_to_maximize, options)



hold on;
for t = 2:q
    Q_curr = value(Qs(t));
    [U,S,V] = svd(Q_curr);
    plot(1:N, U(:,1));
end
hold off; 