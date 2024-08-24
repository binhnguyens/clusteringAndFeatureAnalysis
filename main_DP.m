

%% Run the wavelet filters
% wv_convo;
% wv_noise;
% wv_walk;

%% Clear and reset
clear;
clc;

%% PHQ scores - pre and post

[user_1, phq_1] = phq_pre_post ('phq_pre.txt');
[user_2, phq_2] = phq_pre_post ('phq_post.txt');

phq_values = {}; 
phq_values{1} = user_1;
phq_values{2} = user_2;
phq_values{3} = phq_1;
phq_values{4} = phq_2;

clearvars -except phq_values

%% PHQ label
n_subjects = 38;
% phq_label = zeros (n_subjects, 2);
% 
% for i = 1:length (phq_values{2}) 
%     
%     location = find (phq_values{1} == phq_values{2}(i));
%     phq_label (i,1) = phq_values{3} (location);
%     phq_label (i,2) = phq_values{4} (i);
%     
% end
% 
% phq_label;

phq_label = phq9('phq_score.txt');

%% Trend view

trend_view = zeros (n_subjects, 12); % 38x12 because 12 features (4 per trend)
trend_view(:,1:4) = trend_extract ('mat_walk.txt');
trend_view(:,5:8) = trend_extract ('mat_noise.txt');
trend_view(:,9:12) = trend_extract ('mat_convo.txt');

%% Text 2 array - loading the avg and loc matrices
avg_view = text2array ('avg_view.txt',12);
loc_view = text2array ('loc_view.txt',9);


%% For classification learner
df = [avg_view loc_view trend_view phq_label];
    
%% TSNE

y_tsne = tsne(df);
gscatter(y_tsne(:,1),y_tsne(:,2),phq_values{2})





