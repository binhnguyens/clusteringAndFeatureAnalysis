% Process PANAS and PSS
% October 23, 2023

clear; clc;
addpath ('/Users/binhnguyen/Desktop/Desktop/Digital Mental Health/2. Data and Analysis/Dartmouth dataset/survey');

PSS = readtable('PerceivedStressScale.csv');

% remove 25 and 34
PSS([25,34],:) = [];


PSS_arr = table2array(PSS);
PSS_arr = strrep(PSS_arr, 'Never','0');
PSS_arr = strrep(PSS_arr, 'Almost never','1');
PSS_arr = strrep(PSS_arr, 'Sometime','2');
PSS_arr = strrep(PSS_arr, 'Fairly often','3');
PSS_arr = strrep(PSS_arr, 'Very often','4');
PSS_sum = sum (str2double (PSS_arr(:,3:end)),2);
% PSS_arr = join (cell2table (PSS_arr(:,1)), array2table(PSS_sum))


pre = PSS_arr (1:44, :);
post = PSS_arr (45:end, :);


