% Parameters
q = 89;
n = q * q;                 % Signal length
eps = 0;                   % Recovery tolerance (ideal case, no noise)
k_vals = [20];          % Sparsity levels

fprintf("===== Comparison of Array Code vs Gaussian Matrices =====\n");

for idx = 1:length(k_vals)
    k = k_vals(idx);
    d_L = k + 1;
    m = d_L * q;

    fprintf("\n--- For k = %d ---\n", k);

    % Generate k-sparse signal
    x = zeros(n, 1);
    support = randperm(n, k);
    x(support) = randn(k, 1);

    %% ==== ARRAY LDPC MATRIX ====
    H = generate_array_ldpc_matrix(q, d_L);
    A_bin = H(1:m, 1:n);
    y_bin = A_bin * x;

    tic;
    cvx_begin quiet
        variable x_bin_rec(n)
        minimize(norm(x_bin_rec, 1))
        subject to
            norm(A_bin * x_bin_rec - y_bin, 2) <= eps
    cvx_end
    time_bin = toc;

    %% ==== GAUSSIAN RANDOM MATRIX ====
    A_gauss = randn(m, n) / sqrt(m);
    y_gauss = A_gauss * x;

    tic;
    cvx_begin quiet
        variable x_gauss_rec(n)
        minimize(norm(x_gauss_rec, 1))
        subject to
            norm(A_gauss * x_gauss_rec - y_gauss, 2) <= eps
    cvx_end
    time_gauss = toc;

    %% ==== Report ====
    fprintf("Array Code Matrix Recovery Time:   %.4f seconds\n", time_bin);
    fprintf("Gaussian Matrix Recovery Time:     %.4f seconds\n", time_gauss);

    %% ==== Visualization ====
    figure('Name', sprintf('Signal Recovery for k = %d', k), 'NumberTitle', 'off');
    
    % Original signal
    subplot(3, 1, 1);
    stem(x, 'Marker', 'none');
    title(sprintf('Original Signal (k = %d)', k));
    xlabel('Index');
    ylabel('Amplitude');
    grid on;

    % Reconstructed from Array Code
    subplot(3, 1, 2);
    stem(x_bin_rec, 'r', 'Marker', 'none');
    title('Reconstructed Signal - Array Code');
    xlabel('Index');
    ylabel('Amplitude');
    grid on;

    % Reconstructed from Gaussian
    subplot(3, 1, 3);
    stem(x_gauss_rec, 'g', 'Marker', 'none');
    title('Reconstructed Signal - Gaussian');
    xlabel('Index');
    ylabel('Amplitude');
    grid on;
end

%% --- Structured Array LDPC Matrix Generator ---
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
