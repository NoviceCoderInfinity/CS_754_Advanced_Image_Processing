function success = alm_rpca(r,fs, flag)
%function to implemented RPCA using Augmented Lagrangian Method (ALM)

%generate matrix M = L + S
n1 = 800;
n2 = 900;
s = round(fs * n1 * n2);

%generate low-rank matrix L using truncated SVD method : product of
%low-rank gaussian random matrics
A = randn(n1, r);
B = randn(n2, r);
L = A * B'; %size of L is n1xn2

%generate sparse matrix S with s non-zero elements drawn from N(0, 9)
S = zeros(n1, n2);
idx = randperm(n1 * n2, s); %the non-zero elements occur at random indices, so take a random permutations idx
S(idx) = sqrt(9) * randn(s, 1);

%construct ground truth matrix M
M = L + S;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Now, we will do the ALM algo for RPCA

%Note that I've used mu = 0.25*n1*n2/L1_norm(M) which is fixed as done in the refernced paper ( mu here is a parameter used in the pseudo-code of ALM given
%in slides(page 80 of "Matrix-Recovery")

mu_0 = 1.25 / (norm(M, 2)); % Initial mu value
mu_bar = 1e6;              % Maximum mu value
rho = 1.5;                 % Mu update parameter
lambda = 1 / sqrt(max(n1, n2));

%initialize variables
S_hat = zeros(n1, n2);
L_hat = zeros(n1, n2);
Y = zeros(n1, n2);

%soft-thresholding function
soft_thresh = @(x, tau) sign(x) .* max(abs(x) - tau, 0);

%convergence parameters
max_iter = 500;
eps = 1e-6;
mu = mu_0;

for k = 1:max_iter
    % Update L_hat using singular value thresholding
    [U, Sigma, V] = svd(M - S_hat + (1/mu)*Y, 'econ');
    Sigma_thresh = soft_thresh(diag(Sigma), 1/mu);
    L_hat = U * diag(Sigma_thresh) * V';

    % Update S_hat using soft thresholding
    temp = M - L_hat + (1/mu)*Y;
    S_hat = soft_thresh(temp, lambda/mu);

    % Update Lagrange multiplier
    Y = Y + mu * (M - L_hat - S_hat);
    mu = min(rho * mu, mu_bar);

    % Check convergence
     err = norm(M - L_hat - S_hat, 'fro') / norm(M, 'fro');
     if err < eps
         break;
     end
end


% Step 3: Check reconstruction success
rel_err_L = norm(L - L_hat, 'fro') / norm(L, 'fro');
rel_err_S = norm(S - S_hat, 'fro') / norm(S, 'fro');

success = double(rel_err_L <= 0.001 && rel_err_S <= 0.001);
if (flag == 1) %plot the reconstructed matrices
    figure;
    subplot(2,2,1);
    imagesc(L); colormap gray; colorbar;
    title('Ground Truth L');
    
    subplot(2,2,2);
    imagesc(S); colormap gray; colorbar;
    title('Ground Truth S');
    
    subplot(2,2,3);
    imagesc(L_hat); colormap gray; colorbar;
    title('Estimated L\_hat');
    
    subplot(2,2,4);
    imagesc(S_hat); colormap gray; colorbar;
    title('Estimated S\_hat');
end



end