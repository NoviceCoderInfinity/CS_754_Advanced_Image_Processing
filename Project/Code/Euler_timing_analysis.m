% Parameters
q = 89;                         % Prime number for Euler matrix
n = q * q;                      % Signal length = q^2
eps = 0;                        % Recovery tolerance (noiseless case)
k_vals = [20];               % Sparsity levels

fprintf("===== Euler Square Matrix Recovery Performance =====\n");

for idx = 1:length(k_vals)
    k = k_vals(idx);
    d_L = k + 1;                % Column weight
    m = d_L * q;                % Number of measurements

    fprintf("\n--- For k = %d ---\n", k);

    % Generate k-sparse signal
    x = zeros(n, 1);
    support = randperm(n, k);
    x(support) = randn(k, 1);

    %% ==== EULER SQUARE MATRIX ====
    A_euler = generate_euler_matrix(q, d_L);
    y_euler = A_euler * x;

    tic;
    cvx_begin quiet
        variable x_euler_rec(n)
        minimize(norm(x_euler_rec, 1))
        subject to
            norm(A_euler * x_euler_rec - y_euler, 2) <= eps
    cvx_end
    time_euler = toc;

    %% ==== SSIM Evaluation ====
    dim = sqrt(n);
    x_2D = reshape(x, dim, []);
    x_euler_2D = reshape(x_euler_rec, dim, []);
    ssim_euler = ssim(x_euler_2D, x_2D);

    %% ==== Reporting ====
    fprintf("Euler Matrix Recovery Time:    %.4f seconds\n", time_euler);
    fprintf("Euler Matrix SSIM:             %.4f\n", ssim_euler);

    %% ==== Visualization ====
    figure('Name', sprintf('Euler Recovery (k = %d)', k), 'NumberTitle', 'off');

    subplot(2, 1, 1);
    stem(x, 'Marker', 'none');
    title(sprintf('Original Signal (k = %d)', k));
    xlabel('Index');
    ylabel('Amplitude');
    grid on;

    subplot(2, 1, 2);
    stem(x_euler_rec, 'g', 'Marker', 'none');
    title(sprintf('Reconstructed - Euler (SSIM = %.4f)', ssim_euler));
    xlabel('Index');
    ylabel('Amplitude');
    grid on;
end

%% --- Euler Square Matrix Generator ---
function A = generate_euler_matrix(q, l)
    if l >= q
        error('l must be less than q');
    end

    A = zeros(l*q, q^2);

    for i = 0:q-1
        for j = 0:q-1
            col = i * q + j + 1;
            for k = 0:l-1
                row = mod(i * k + j, q) + k * q + 1;
                A(row, col) = 1;
            end
        end
    end
end
