n = 961;                    % Signal dimension (e.g., 31*31)
q = 31;                     % Prime such that n = q^2
eps = 1e-6;                 % Recovery tolerance
trials = 10;                 % Trials per (m, k)
success_map = zeros(20, 20); % Preallocate success grid

fprintf("=== Starting Phase Transition Simulation ===\n");

m_vals = linspace(100, 900, 9);  % Example measurement range
for i = 1:length(m_vals)
    m = round(m_vals(i));
    for k = 5:(m/20):m
        d_L = ceil(m / q);
        if d_L >= q
            fprintf('Skipping m=%d, k=%d: d_L=%d ≥ q\n', m, k, d_L);
            continue;
        end

        success_count = 0;

        for t = 1:trials
            % Generate k-sparse signal
            x = zeros(n, 1);
            support = randperm(n, k);
            x(support) = randn(k, 1);

            % Generate LDPC sensing matrix
            % H = generate_array_ldpc_matrix(q, d_L);
            %if size(H,1) < m
            %    continue;
            %end
            %A = H(1:m, 1:n);
            A = randn(m, n) / sqrt(m);
            y = A * x;

            % ℓ1 minimization
            try
                cvx_begin quiet
                    variable x_rec(n)
                    minimize(norm(x_rec, 1))
                    subject to
                        norm(A * x_rec - y, 2) <= eps
                cvx_end

                if norm(x - x_rec) / norm(x) < 1e-2
                    success_count = success_count + 1;
                end
            catch
                continue;  % Skip on solver failure
            end
        end

        success_rate = success_count / trials;
        success_map(j, i) = success_rate;
        fprintf("m = %d, k = %d, success = %.2f\n", m, k, success_rate);
    end
end

[k_grid, m_grid] = meshgrid(k_vals, m_vals);
x_axis = m_grid / n;        % m/n
y_axis = k_grid ./ m_grid;  % k/m

figure;
imagesc(x_axis(1,:), y_axis(:,1), success_map');  % Transpose map!
axis xy;
xlabel('m / n (measurement rate)');
ylabel('k / m (sparsity density)');
title('Phase Transition Diagram (LDPC Matrix)');
colorbar;
colormap(jet);


function H = generate_array_ldpc_matrix(q, l)
    if ~isprime(q)
        error('q must be prime.');
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
            exponent = mod((i - 1)*(j - 1), q);
            block = P^exponent;
            H((i-1)*q+1:i*q, (j-1)*q+1:j*q) = block;
        end
    end
end
