clc; clear; close all;

% Load the cryoEM image
img = im2double(imread('cryoem.png'));
[Nx, Ny] = size(img);

% Set the values of N
N_vals = [50, 100, 500, 1000, 2000, 5000, 10000];

for N = N_vals
    % Generate random angles in [0, 360] degrees
    angles = 360 * rand(1, N);
    
    % Compute Radon transform
    [R, xp] = radon(img, angles);
    
    % Generate reversed projections
    angles_reversed = mod(angles + 180, 360);
    [R_reversed, ~] = radon(img, angles_reversed);
    
    % Combine original and reversed projections
    angles_aug = [angles, angles_reversed];
    R_aug = [R, R_reversed];
    
    % Construct similarity graph (binary k-NN graph)
    k = 5; % Number of neighbors
    D = pdist2(R_aug', R_aug');
    [~, idx] = sort(D, 2);
    W = zeros(size(D));
    for i = 1:size(D, 1)
        W(i, idx(i, 2:k+1)) = 1;
    end
    W = max(W, W'); % Make symmetric
    
    % Compute graph Laplacian
    D = diag(sum(W, 2));
    L = D - W;
    
    % Solve generalized eigenvalue problem
    [eigVectors, eigValues] = eig(L, D);
    eigValues = diag(eigValues);
    [eigValues, order] = sort(eigValues);
    eigVectors = eigVectors(:, order);

    % Extract the first two non-trivial eigenvectors
    phi1 = eigVectors(:, 2);
    phi2 = eigVectors(:, 3);

    % Compute estimated angles
    theta_est = atan2(phi2, phi1);
    [~, sort_idx] = sort(theta_est);
    angles_sorted = angles_aug(sort_idx);
    
    % Reconstruct image using sorted projections
    img_recon = iradon(R_aug(:, sort_idx), angles_sorted, 'linear', 'Ram-Lak', 1, Nx);
    
    % Optimize rotation for best alignment
    min_rmse = inf;
    best_img = img_recon;
    for rot_angle = 0:1:360
        img_rotated = imrotate(img_recon, rot_angle, 'crop');
        rmse = sqrt(mean((img(:) - img_rotated(:)).^2));
        if rmse < min_rmse
            min_rmse = rmse;
            best_img = img_rotated;
        end
    end
    
    % Save reconstructed image with appropriate filename
    filename = sprintf('reconstructed_N_%d.png', N);
    imwrite(best_img, filename);
    
    % Display results
    figure;
    subplot(1, 2, 1); imshow(img, []); title('Original Image');
    subplot(1, 2, 2); imshow(best_img, []); title(['Reconstructed Image, N = ', num2str(N)]);
    fprintf('N = %d, RMSE = %.4f\n', N, min_rmse);
end
