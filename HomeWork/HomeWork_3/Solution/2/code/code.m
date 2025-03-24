% Adds the MMread folder as a function
addpath('./MMread/');

seed = 0;
% threshold = 1e-3;
threshold = 0.1;
dimensions = [120 240];
patch_size = 8;
RMSE_info = "";
videos = ["cars" "flame"];
start_frames = dictionary(videos, [1 3]);
start_heights = dictionary(videos, [169 115]);
start_widths = dictionary(videos, [113 65]);
T = [3 5 7];

for video = videos
    create_directory("../images/"+video);
    for t = T
        RMSE_info = process(video, t, dimensions, patch_size, seed, threshold, RMSE_info, start_heights, start_widths, start_frames);
    end
end
file = fopen('../images/RMSE.txt', 'wt');
fprintf(file, RMSE_info);
fclose(file);

function RMSE_info = process(image_name, T, dimensions, patch_size, seed, threshold, RMSE_info, start_heights, start_widths, start_frames)
    rng(seed);

    disp(image_name+', T = '+string(T));
    video_input = mmread(char("../images/"+ image_name +".avi"));
    start_frame = start_frames(image_name);
    for i=1:T
        X(:,:,i) = double(rgb2gray(video_input.frames(i+start_frame-1).cdata));
    end
    % cropping
    H = dimensions(1);
    W = dimensions(2);
    X = X(start_heights(image_name):start_heights(image_name)+H-1, start_widths(image_name):start_widths(image_name)+W-1, :);
    code = randi([0 1], size(X));
    coded_snapshot = X.*code;
    coded_snapshot = sum(coded_snapshot, 3) + generate_gaussian_noise(dimensions, 0, 4);

    for i=1:T
        save_image(X(:,:,i), "../images/"+image_name+"/"+"frame = "+string(i)+".png")
    end
    save_image(coded_snapshot/T, "../images/"+image_name+"/"+"coded snapshot, T = "+string(T)+".png");
    save_image(coded_snapshot./(max(sum(code, 3), ones(dimensions))), "../images/"+image_name+"/"+"coded snapshot weighted averaging, T = "+string(T)+".png");

    dct_matrix = kron(dctmtx(patch_size)',dctmtx(patch_size)');
    dct_matrix = repmat({dct_matrix}, T, 1);
    dct_matrix = blkdiag(dct_matrix{:});
    % dct_matrix = kron(dct_matrix, dctmtx(T)');
    image_reconstructed = patch_reconstruct(coded_snapshot, patch_size, code, dct_matrix, threshold);
    for i=1:T
        save_image(image_reconstructed(:,:,i), "../images/" + image_name + "/reconstructed, T = " + string(T) + ", frame = " + string(i) +".png")
    end
    RMSE = calculate_RMSE(X, image_reconstructed);
    RMSE_info = RMSE_info + "RMSE = " + string(RMSE) + " for " + image_name + " reconstructed, T = " + string(T) + newline;
end

function image_reconstructed = patch_reconstruct(coded_snapshot, patch_size, code, dct_matrix, threshold)
    dimensions = size(code);
    H = dimensions(1);
    W = dimensions(2);
    T = dimensions(3);
    number_of_overlapping_patches = zeros(dimensions);
    image_reconstructed = zeros(dimensions);
    for i = 1:H+1-patch_size
        for j = 1:W+1-patch_size
            x = coded_snapshot(i:i+patch_size-1, j:j+patch_size-1);
            c = code(i:i+patch_size-1, j:j+patch_size-1,:);
            c = reshape(c, patch_size*patch_size, T);
            phi = cell(1,T);
            for k=1:T
                phi{k} = diag(c(:,k));
                % phi{k} = spdiags(c(:,k), 0, H*W, H*W); % use sparse matrices for large patches
            end
            measurement_matrix = cat(2, phi{:});
            theta_hat = ista(x(:),  measurement_matrix*dct_matrix, threshold, 25);
            x_hat = dct_matrix * theta_hat;
            image_reconstructed(i:i+patch_size-1, j:j+patch_size-1, :) = image_reconstructed(i:i+patch_size-1, j:j+patch_size-1, :) + reshape(x_hat, [patch_size, patch_size, T]);
            number_of_overlapping_patches(i:i+patch_size-1, j:j+patch_size-1, :) = number_of_overlapping_patches(i:i+patch_size-1, j:j+patch_size-1, :) + 1;
        end
    end
    image_reconstructed = image_reconstructed ./ number_of_overlapping_patches;
end

function gaussian_noise_matrix = generate_gaussian_noise(size, mean, variance)
    gaussian_noise_matrix = sqrt(variance)*(mean + randn(size));
end

function RMSE = calculate_RMSE(image_original, image_reconstructed)
    RMSE = norm(image_reconstructed(:) - image_original(:)) / norm(image_original(:));
end

function image_input = image_bound(image_input)
    image_input(image_input < 0) = 0;
    image_input(image_input > 255)= 255;
end

function image_integer = image_double_to_integer(image_double)
    image_integer = uint8(image_double);
end

function [] = save_image(image, filename)
    image = image_bound(image);
    image = image_double_to_integer(image);
    imwrite(image, filename);
end

function theta = ista(y, A, threshold, lamdba)
    theta = randn([size(A,2) 1]);
    % generates a random vector sampled from standard Gaussian with size
    % Number of columns of A by 1
    alpha = eigs(A'*A,1);
    % d=eigs(A,k) return k largest magnitude eigenvalues
    while(true)
        theta_next = wthresh(theta + 1/alpha*A'*(y-A*theta), 's', lamdba / (2*alpha));
        if norm(theta_next-theta) < threshold
            break
        end
        theta = theta_next;
    end
end