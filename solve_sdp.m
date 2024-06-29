N = 6;
q = 6;

Rs = {eye(N)};
for i = 1:N
    Rs{i+1} = tril(circshift(Rs{i}, -1, 2));
end

% Setting up the SDP
J = fliplr(eye(N/2));
E = ones(N/2, N/2);

As = cell(1,q+1);
Bs = cell(1,q+1);

As{1} = E / N;
As{q} = eye(N/2)/N;

Bs{1} = E / N;
Bs{q} = zeros(N/2, N/2);

Constraints = cell(1, (N + 3)*q); % 4q + (N-1)q 
cc = 1;

for i = 1:q
    As{i} = sdpvar(N/2, N/2);
    Bs{i} = sdpvar(N/2, N/2, 'full');

    Constraints{cc} = (trace(As{i}) == 0.5);
    Constraints{cc+1} = (J * Bs{i} * J == transpose(Bs{i}));

    Constraints{cc+2} = ((As{i} + Bs{i}) * J >= 0);
    Constraints{cc+3} = ((As{i} - Bs{i}) * J >= 0);
    cc = cc + 4;
end

for t = 2:(q+1)
    Q_curr = [As{t}, Bs{t}; J * Bs{t} * J, J * As{t} * J];
    Q_prev = [As{t-1}, Bs{t-1}; J * Bs{t-1} * J, J * As{t-1} * J];

    for i=1:N 
        Constraints{cc} = %TODO
        cc = cc + 1;
    end 
end 