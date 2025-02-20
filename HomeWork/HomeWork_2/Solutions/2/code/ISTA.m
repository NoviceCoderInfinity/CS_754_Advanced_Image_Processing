clc; clear; close all;

%% Load and preprocess images
img1 = double(imread('barbara256.png'));
img2 = double(imread('goldhill.png'));
img2 = img2(1:256, 1:256); % Crop top-left 256x256 portion

% Add Gaussian noise (mean 0, variance 4)
sigma = 2; % sqrt(variance)
noisy_img1 = img1 + sigma * randn(size(img1));
noisy_img2 = img2 + sigma * randn(size(img2));

% Save noisy images
imwrite(uint8(noisy_img1), 'Noisy_Barbara.png');
imwrite(uint8(noisy_img2), 'Noisy_Goldhill.png');

% ISTA parameters
lambda = 1;
patch_size = 8;
M = 32; N = 64; % Measurement matrix size

% Generate Gaussian measurement matrix
Phi = randn(M, N);

% Compute 2D-DCT basis
DCT_basis = kron(dctmtx(patch_size), dctmtx(patch_size));

% ISTA Reconstruction function
function theta = ista(y, A, lambda)
    theta = randn(size(A, 2), size(y, 2));
    alpha = eigs(A' * A, 1);
    prev_norm = 0;
    current_norm = norm(y - A * theta, "fro");
    
    while abs(prev_norm - current_norm) > 0.001
        prev_norm = norm(y - A * theta, "fro");
        theta = wthresh(theta + (A' / alpha) * (y - A * theta), "s", lambda / (2 * alpha));
        current_norm = norm(y - A * theta, "fro");
    end
end

% Function to reconstruct image
function X_recon = reconstruct_image(X, Phi, DCT_basis, lambda)
    [H, W] = size(X);
    X_recon = zeros(H, W);
    X_patches = im2col(X, [8, 8]);
    patch_recon_count = zeros(size(X_recon));
    [~, num_patches] = size(X_patches);
    
    Y = Phi * X_patches;
    X_patches_recon = DCT_basis * ista(Y, Phi * DCT_basis, lambda);
    
    for patch_idx = 1:num_patches
        patch = X_patches_recon(:, patch_idx);
        patch = reshape(patch, [8, 8]);
        
        row = mod(patch_idx - 1, H - 7) + 1;
        col = floor((patch_idx - 1) / (H - 7)) + 1;
        
        X_recon(row:row+7, col:col+7) = X_recon(row:row+7, col:col+7) + patch;
        patch_recon_count(row:row+7, col:col+7) = patch_recon_count(row:row+7, col:col+7) + 1;
    end
    
    X_recon = X_recon ./ patch_recon_count;
    X_recon = 255 * (X_recon - min(X_recon, [], "all")) / (max(X_recon, [], "all") - min(X_recon, [], "all"));
end

% Reconstruct images
recon_img1 = reconstruct_image(noisy_img1, Phi, DCT_basis, lambda);
recon_img2 = reconstruct_image(noisy_img2, Phi, DCT_basis, lambda);

% Save reconstructed images
imwrite(uint8(recon_img1), 'Reconstructed_Barbara_ISTA.png');
imwrite(uint8(recon_img2), 'Reconstructed_Goldhill_ISTA.png');

% Compute RMSE
rmse1 = norm(img1(:) - recon_img1(:)) / norm(img1(:));
rmse2 = norm(img2(:) - recon_img2(:)) / norm(img2(:));

% Display results
figure;
subplot(2,3,1), imshow(img1, [0, 255]), title('Original Barbara');
subplot(2,3,2), imshow(noisy_img1, [0, 255]), title('Noisy Barbara');
subplot(2,3,3), imshow(recon_img1 / 255, []), title(['Reconstructed Barbara (RMSE=', num2str(rmse1), ')']);
subplot(2,3,4), imshow(img2, [0, 255]), title('Original Goldhill');
subplot(2,3,5), imshow(noisy_img2, [0, 255]), title('Noisy Goldhill');
subplot(2,3,6), imshow(recon_img2 / 255, []), title(['Reconstructed Goldhill (RMSE=', num2str(rmse2), ')']);
