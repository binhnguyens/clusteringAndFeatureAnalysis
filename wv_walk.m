%% Resources used
% http://www.numerical-tours.com/matlab/denoisingwav_1_wavelet_1d/
% https://cagnazzo.wp.imt.fr/files/2018/04/tp_sd205.pdf
% https://blancosilva.wordpress.com/teaching/mathematical-imaging/denoising-wavelet-thresholding/

clc;
clear;
close all;
addpath ('/Users/binhnguyen/Documents/MATLAB/Data Preparation')

%% Load file 
file1 = 'python_walk.txt'
signal = importdata (file1);

index = find (signal == -1000);
num_subjects = length (index);
num_days = 66;
matrix = zeros (num_subjects, num_days);

matrix(1,:) = transpose (signal(1:index (1)-1));
for i = 2:length(index) % 1 to 38 (Number of subjects)
    end_index = length ((signal(1+index(i-1):index (i)-1)));
    matrix(i,1:end_index) = transpose (signal(1+index(i-1):index (i)-1));
end 



%% Big Loop
big_b = zeros(num_subjects,4);

for sig_num = 1:num_subjects % Start big loop
    
    %% Signal assignment
    e = matrix(sig_num,:); % Wavelet signal
    sig = e; % Original signal
    % figure;plot(e); 
    
    
    %% Wavelet denoise
    w = 'haar';
    n = 10;
    [C,L] = wavedec(e,n,w);
    for i = 1:n
        A(i,:) = wrcoef('a',C,L,w,i);
        D(i,:) = wrcoef('d',C,L,w,i);
    end
    e=e-A(10,:);
    e=e-(D(2,:)+D(1,:));
    e = e*2; %Get the signal to be at the same ampltiude by multiplying by two
    
    
    %% Least squares from forum
    % https://www.mathworks.com/matlabcentral/answers/36999-how-do-i-regression-fit-a-sinwave-to-a-dataset
    % https://www.mathworks.com/help/stats/fitnlm.html#d122e353492
    Y = e;
    X = 1:length (e);
    
    modelfun = @(b,x)(b(1) + b(2)*sin(x*b(3) + b(4)));
    
    B1 = trimmean(Y,10);  % Vertical shift
    B2 = (max(Y) - min(Y))/2; % Amplitude
    pks = findpeaks(diff(e),num_days,'MinPeakHeight',10,'MinPeakDistance',0.1);
    B3 = 2*pi/(num_days/length(pks)); % Phase (Number of peaks)
    B4 = 0; % Phase shift (eyeball the Curve)
    
    beta0 = [B1 B2 B3 B4];
    mdl = fitnlm(X,Y,modelfun,beta0);
    
    bitch = mdl.Coefficients.Estimate;
    t = num2cell(bitch);
    [big_b(sig_num,1),big_b(sig_num,2),big_b(sig_num,3),big_b(sig_num,4)] = ...
        deal(t{:});
    
    
    %% Plotting 
    figure;
    x = 1:length (e);
    plot(x, sig, 'c', x,e,'b',  X, mdl.Fitted, 'r' )
    legend ('Original','Wavelet Denoised','Least square sinusoidal regression')
    title ('Daily walk activity duration')
    xlabel ('Day')
    ylabel ('Seconds')
%     grid

end % End big loop

%% Export file to Python
fid = fopen('mat_walk.txt','wt');
fprintf(fid,'%.2f\n',big_b);
fclose(fid);
    
