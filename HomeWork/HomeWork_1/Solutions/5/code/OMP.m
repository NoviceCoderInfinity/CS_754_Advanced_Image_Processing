image_size = 32;
algo_name = "OMP";
set(0, 'DefaultFigureVisible', 'off');

k_values = [5, 10, 20, 30, 50, 100, 150, 200];
m_values = 100:100:1000;

plot_k_idxs = [1,5,8];
plot_m_idxs = [5,7];

RMSE = zeros(length(k_values), length(m_values));
ORIG_IMAGES = zeros(image_size, image_size, length(k_values));

% RECONSTRUCTIONS
RECON_IMAGES_P1 = zeros(image_size, image_size, length(m_values), length(plot_k_idxs));
RECON_IMAGES_P2 = zeros(image_size, image_size, length(k_values), length(plot_m_idxs));

% ITERATIONS
for idx_k = 1:length(k_values)
    k = k_values(idx_k);

    coefficients = zeros(image_size^2, 1);
    selected_indices = randperm(image_size^2, k);
    coefficients(selected_indices) = randn(k, 1);
    psi = dctmtx(image_size^2);
    vec_f = psi * coefficients;
    f = reshape(vec_f, image_size, image_size);

    ORIG_IMAGES(:, :, idx_k) = f;
    
    for idx_m = 1:length(m_values)
        m = m_values(idx_m);
        phi = randi([0, 1], m, image_size^2);
        phi(phi == 0) = -1;
        A = phi * psi;
        y = phi * vec_f;

        theta = CoSAMP(A, y, k, 1000);
        f_hat = reshape(psi*theta, image_size, image_size);
        
        if any(plot_k_idxs == idx_k)
            RECON_IMAGES_P1(:, :, idx_m, find(plot_k_idxs == idx_k)) = f_hat;
        end
        if any(plot_m_idxs == idx_m)
            RECON_IMAGES_P2(:, :, idx_k, find(plot_m_idxs == idx_m)) = f_hat;
        end
        
        RMSE(idx_k, idx_m) = norm(vec_f - f_hat(:)) / norm(vec_f);
    end
end

% PLOTS

for i = 1:length(k_values)
    z = ORIG_IMAGES(:,:,i);
    imwrite(mat2gray(z), sprintf("./orig_%s_k=%d.png", algo_name, k_values(i)));
end

for i = 1:length(plot_k_idxs)
    for j = 1:length(m_values)
        z = RECON_IMAGES_P1(:,:,j,i);
        imwrite(mat2gray(z), sprintf("./recon_p1_%s_k=%d_m=%d.png", algo_name, k_values(plot_k_idxs(i)), m_values(j)));
    end
end

for i = 1:length(plot_m_idxs)
    for j = 1:length(k_values)
        z = RECON_IMAGES_P2(:,:,j,i);
        imwrite(mat2gray(z), sprintf("./recon_p2_%s_k=%d_m=%d.png", algo_name, k_values(j), m_values(plot_m_idxs(i))));
    end
end

figure;
hold on;
legends = [];
for idx = 1:length(plot_k_idxs)
    k_idx = plot_k_idxs(idx);
    k = k_values(k_idx);

    y_axis = RMSE(k_idx,:);
    x_axis = m_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("k=%d",k)];
end
xlabel('m');
ylabel('RMSE');
title(sprintf("RMSE vs m for varying k - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("./%s_k.png", algo_name));

figure;
hold on;
legends = [];
for idx = 1:length(plot_m_idxs)
    m_idx = plot_m_idxs(idx);
    m = m_values(m_idx);

    y_axis = RMSE(:,m_idx);
    x_axis = k_values;
    plot(x_axis, y_axis);
    legends = [legends, sprintf("m=%d",m)];
end
xlabel('k');
ylabel('RMSE');
title(sprintf("RMSE vs k for varying m - %s", algo_name));
legend(legends);
hold off;
saveas(gcf, sprintf("./%s_m.png", algo_name));

function res = CoSAMP(Phi, u, K, max_iter)
    % CoSAMP algorithm for sparse signal recovery
    % Input:
    %   Phi - measurement matrix (m x n)
    %   u - observed signal (m x 1)
    %   K - sparsity level (number of non-zero coefficients to retain)
    %   max_iter - maximum number of iterations
    % Output:
    %   res - the sparse signal estimate (n x 1)

    % Initialization
    n = size(Phi, 2);
    Sest = zeros(n, 1);  % The estimated sparse signal
    v = u;  % Residual is initially the observed signal
    T = [];  % Support set (indices of selected components)
    error = norm(v) / norm(u);  % Initial error between residual and original signal

    for t = 1:max_iter
        % Step 1: Identification
        % Compute the proxy as the absolute value of the correlations between residual and columns of Phi
        proxy = abs(Phi' * v);
        [vals, idx_new] = sort(proxy, 'descend');
        Omega = idx_new(1:2*K);  % Select top 2K largest components
        T = union(T, Omega);  % Merge new components with current support set
        
        % Step 2: Estimation (Least squares over the merged support set)
        theta = zeros(n, 1);
        theta(T) = pinv(Phi(:, T)) * u;  % Least-squares solution

        % Step 3: Pruning
        % Retain only the largest K entries in theta
        [~, idx_prune] = sort(abs(theta(T)), 'descend');
        T = T(idx_prune(1:K));  % Keep top K coefficients

        % Update the sparse estimate
        Sest = zeros(n, 1);
        Sest(T) = theta(T);
        
        % Step 4: Sample Update (Update residual)
        v = u - Phi(:, T) * theta(T);

        % Step 5: Check for convergence (relative error)
        error = norm(v) / norm(u);
    end

    % Return the final sparse signal estimate
    res = Sest;
end
