%this code uses the alm_rpca function to run the rpca algo 15 times for
%each (r,fs) pair value

r_vals = [75, 100, 125, 150, 200];
fs_vals = [0.04, 0.06, 0.08, 0.1, 0.15];
num_trials = 5;

success_map = zeros(length(fs_vals), length(r_vals));

for i = 1:length(fs_vals)
    for j = 1:length(r_vals)
        success_count = 0;
        for trial = 1:num_trials
            success = alm_rpca(r_vals(j), fs_vals(i),0);
            success_count = success_count + success;
        end
        success_map(i, j) = success_count / num_trials;
    end
end

imagesc(r_vals, fs_vals, success_map);
xlabel('Rank r');
ylabel('Sparsity fraction f_s');
title('Success Probability of ALM-RPCA');
colorbar;
colormap(gray); % darker = lower probability
set(gca, 'YDir', 'normal'); % to ensure fs increases bottom to top
