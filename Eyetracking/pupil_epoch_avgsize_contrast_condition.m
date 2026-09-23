% For subject 1
sub1_table = combinedResults_sub1;

% Average AvgPupilSize over Category and AI
sub1_avgPupilSize = grpstats(sub1_table, {'Category', 'AI'}, 'mean', 'DataVars', 'AvgPupilSize');

% Transpose the table and extract row names
row_names = strcat(sub1_avgPupilSize.Category, '_', sub1_avgPupilSize.AI);
transposed_sub1 = array2table(sub1_avgPupilSize.mean_AvgPupilSize', 'VariableNames', row_names);

% For subject 2
sub2_table = combinedResults_sub2;

% Average AvgPupilSize over Category and AI
sub2_avgPupilSize = grpstats(sub2_table, {'Category', 'AI'}, 'mean', 'DataVars', 'AvgPupilSize');

transposed_sub2= array2table(sub2_avgPupilSize.mean_AvgPupilSize', 'VariableNames', row_names);

% Add subNo column to transposed tables
transposed_sub1.subNo = repmat(1, height(transposed_sub1), 1);
transposed_sub2.subNo = repmat(2, height(transposed_sub2), 1);

% Concatenate transposed tables
concatenated_table = [transposed_sub1; transposed_sub2];

data = concatenated_table;

% Extract the columns
real_p = data.pleasant_NO;
real_n = data.neutral_NO;
real_u = data.unpleasant_NO;
ai_p = data.pleasant_YES;
ai_n = data.neutral_YES;
ai_u = data.unpleasant_YES;

% Calculate means and standard errors
mean_real_p = mean(real_p);
mean_real_n = mean(real_n);
mean_real_u = mean(real_u);
mean_ai_p = mean(ai_p);
mean_ai_n = mean(ai_n);
mean_ai_u = mean(ai_u);

sem_real_p = std(real_p) / sqrt(length(real_p));
sem_real_n = std(real_n) / sqrt(length(real_n));
sem_real_u = std(real_u) / sqrt(length(real_u));
sem_ai_p = std(ai_p) / sqrt(length(ai_p));
sem_ai_n = std(ai_n) / sqrt(length(ai_n));
sem_ai_u = std(ai_u) / sqrt(length(ai_u));

% Perform paired t-tests
[~, p_val_real_p_vs_n] = ttest2(real_p, real_n);
[~, p_val_real_u_vs_n] = ttest2(real_u, real_n);
[~, p_val_ai_p_vs_n] = ttest2(ai_p, ai_n);
[~, p_val_ai_u_vs_n] = ttest2(ai_u, ai_n);

% Define significance label logic inline
% ... (same code as before) ...
% Create the figure

% Define significance label logic inline
if p_val_real_p_vs_n < 0.001
    sig_label_real_p_vs_n = '***';
elseif p_val_real_p_vs_n < 0.01
    sig_label_real_p_vs_n = '**';
elseif p_val_real_p_vs_n < 0.05
    sig_label_real_p_vs_n = '*';
else
    sig_label_real_p_vs_n = 'n.s.';
end

if p_val_real_u_vs_n < 0.001
    sig_label_real_u_vs_n = '***';
elseif p_val_real_u_vs_n < 0.01
    sig_label_real_u_vs_n = '**';
elseif p_val_real_u_vs_n < 0.05
    sig_label_real_u_vs_n = '*';
else
    sig_label_real_u_vs_n = 'n.s.';
end

if p_val_ai_p_vs_n < 0.001
    sig_label_ai_p_vs_n = '***';
elseif p_val_ai_p_vs_n < 0.01
    sig_label_ai_p_vs_n = '**';
elseif p_val_ai_p_vs_n < 0.05
    sig_label_ai_p_vs_n = '*';
else
    sig_label_ai_p_vs_n = 'n.s.';
end

if p_val_ai_u_vs_n < 0.001
    sig_label_ai_u_vs_n = '***';
elseif p_val_ai_u_vs_n < 0.01
    sig_label_ai_u_vs_n = '**';
elseif p_val_ai_u_vs_n < 0.05
    sig_label_ai_u_vs_n = '*';
else
    sig_label_ai_u_vs_n = 'n.s.';
end

figure;

% Plot 1: Real_P vs Real_N
subplot(2, 2, 1);
bar([1 2], [mean_real_p, mean_real_n]); hold on;
errorbar([1 2], [mean_real_p, mean_real_n], [sem_real_p, sem_real_n], '.k');
title('Real\_P vs Real\_N');
ylabel('Average Pupil Size');
text(1.8,  -0.2, ...
    ['p = ' num2str(p_val_real_p_vs_n, '%.3f') ' ' sig_label_real_p_vs_n], ...
    'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(1, mean_real_p - 4, num2str(mean_real_p, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
text(2, mean_real_n - 4, num2str(mean_real_n, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
set(gca, 'XTick', [1 2], 'XTickLabel', {'Real_P', 'Real_N'});

% Plot 2: Real_U vs Real_N
subplot(2, 2, 2);
bar([1 2], [mean_real_u, mean_real_n]); hold on;
errorbar([1 2], [mean_real_u, mean_real_n], [sem_real_u, sem_real_n], '.k');
title('Real\_U vs Real\_N');
ylabel('Average Pupil Size');
text(1.8,  -0.2, ...
    ['p = ' num2str(p_val_real_u_vs_n, '%.3f') ' ' sig_label_real_u_vs_n], ...
    'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(1, mean_real_u - 0.4, num2str(mean_real_u, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
text(2, mean_real_n - 0.4, num2str(mean_real_n, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
set(gca, 'XTick', [1 2], 'XTickLabel', {'Real_U', 'Real_N'});

% Plot 3: AI_P vs AI_N
subplot(2, 2, 3);
bar([1 2], [mean_ai_p, mean_ai_n]); hold on;
errorbar([1 2], [mean_ai_p, mean_ai_n], [sem_ai_p, sem_ai_n], '.k');
title('AI\_P vs AI\_N');
ylabel('Average Pupil Size');
text(1.8, -0.2, ...
    ['p = ' num2str(p_val_ai_p_vs_n, '%.3f') ' ' sig_label_ai_p_vs_n], ...
    'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(1, mean_ai_p - 0.4, num2str(mean_ai_p, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
text(2, mean_ai_n - 0.4, num2str(mean_ai_n, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
set(gca, 'XTick', [1 2], 'XTickLabel', {'AI_P', 'AI_N'});

% Plot 4: AI_U vs AI_N
subplot(2, 2, 4);
bar([1 2], [mean_ai_u, mean_ai_n]); hold on;
errorbar([1 2], [mean_ai_u, mean_ai_n], [sem_ai_u, sem_ai_n], '.k');
title('AI\_U vs AI\_N');
ylabel('Average Pupil Size');
text(1.8, -0.2, ...
    ['p = ' num2str(p_val_ai_u_vs_n, '%.3f') ' ' sig_label_ai_u_vs_n], ...
    'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
text(1, mean_ai_u - 0.4, num2str(mean_ai_u, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
text(2, mean_ai_n - 0.4, num2str(mean_ai_n, '%.2f'), 'HorizontalAlignment', 'center', 'FontSize', 15);
set(gca, 'XTick', [1 2], 'XTickLabel', {'AI_U', 'AI_N'});

% Adjust layout
sgtitle('Comparison of Pupil Sizes');
set(gcf, 'Position', [100, 100, 1000, 800]);