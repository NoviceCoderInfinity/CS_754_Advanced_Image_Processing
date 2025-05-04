% Fixed parameters
q = 19;                   % Prime number
n = q * q;                % Signal dimension = 961
d_L = 5;                  % Left degree (fixed)
l = d_L;
m = d_L * q;              % Number of measurements = 62
eps = 0;                  % Recovery tolerance
num_trials = 5;          % Number of trials per k

% Range of sparsity levels
k_values = 1:m;
avg_ssim_scores = zeros(size(k_values));

% Generate array code matrix once
H = generate_array_ldpc_matrix(q, l);
%H = generate_euler_matrix(q, l);
%A = H(1:m, 1:n);

% Loop over different sparsity levels
for i = 1:length(k_values)
    k = k_values(i);
    ssim_trials = zeros(num_trials, 1);

    for trial = 1:num_trials
        % Generate random k-sparse signal
        x = zeros(n, 1);
        support = randperm(n, k);
        x(support) = randn(k, 1);

        % Compressed measurements
        A = randn(m, n) / sqrt(m);
        y = A * x;

        % Basis Pursuit via CVX
        cvx_begin quiet
            variable x_rec(n)
            minimize(norm(x_rec, 1))
            subject to
                norm(A * x_rec - y, 2) <= eps
        cvx_end

        % Compute SSIM
        dim = sqrt(n);
        x_mat = reshape(x, dim, []);
        x_rec_mat = reshape(x_rec, dim, []);
        ssim_trials(trial) = ssim(x_rec_mat, x_mat);
    end

    % Average SSIM for this sparsity level
    avg_ssim_scores(i) = mean(ssim_trials);
end

% Plot Average SSIM vs. Sparsity Level
figure;
plot(k_values, avg_ssim_scores, '-o', 'LineWidth', 2);
xlabel('Sparsity Level (k)');
ylabel('Average SSIM over 10 trials');
title(sprintf('Avg SSIM vs. k (Array Code, n = %d, m = %d, d_L = %d)', n, m, d_L));
grid on;

function H = generate_array_ldpc_matrix(q, l)
    if ~isprime(q)
        error('q must be a prime number.');
    end
    if l >= q
        error('l must be less than q.');
    end

    P = zeros(q);
    for i = 1:q
        P(i, mod(i, q) + 1) = 1;
    end

    H = zeros(l * q, q^2);
    for i = 1:l
        for j = 1:q
            exponent = mod((i - 1) * (j - 1), q);
            block = P^exponent;
            row_range = (i - 1)*q + 1 : i*q;
            col_range = (j - 1)*q + 1 : j*q;
            H(row_range, col_range) = block;
        end
    end
end

function A = generate_euler_matrix(q, l)
    % Generate a binary matrix A of size l*q x q^2 with column weight l
    % such that the max inner product between any two columns is <= 1
    % Works for ANY integer q >= 2

    if l >= q
        error('l must be less than q');
    end

    A = zeros(l*q, q^2);  % Preallocate binary matrix

    for i = 0:q-1
        for j = 0:q-1
            col = i * q + j + 1;  % Column index in [1, q^2]
            for k = 0:l-1
                % Compute row index: k * q + (i * k + j) mod q
                row = mod(i * k + j, q) + k * q + 1;
                A(row, col) = 1;
            end
        end
    end
end
