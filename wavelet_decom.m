clc;
clear;
addpath ('/MATLAB Drive/Wavelet Smoothing')

%% Load file 
file1 = 'python_file.txt'
signal = importdata (file1);
e = signal;
samf= length (e);
e = transpose (e);
figure;plot(e); hold on;

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
% figure;
plot(e);hold on;


% %% Least squares subject to sinusoid
% % https://www.mathworks.com/matlabcentral/answers/121579-curve-fitting-to-a-sinusoidal-function
% y = e*2;
% x = 1:length (e);
% 
% % Algorithm taken from website
% yu = max(y);
% yl = min(y);
% yr = (yu-yl);                               % Range of ‘y’
% yz = y-yu+(yr/2);
% zx = x(yz .* circshift(yz,[0 1]) <= 0);     % Find zero-crossings
% per = 2*mean(diff(zx));                     % Estimate period
% ym = mean(y);                               % Estimate offset
% fit = @(b,x)  b(1).*(sin(2*pi*x./b(2) + 2*pi/b(3))) + b(4);    % Function to fit
% fcn = @(b) sum((fit(b,x) - y).^2);                              % Least-Squares cost function
% s = fminsearch(fcn, [yr;  per;  -1;  ym])                       % Minimise Least-Squares
% xp = linspace(min(x),max(x));
% figure;
% plot(x,y,'b',  xp,fit(s,xp), 'r')
% grid


%% Least squares from forum
% https://www.mathworks.com/matlabcentral/answers/36999-how-do-i-regression-fit-a-sinwave-to-a-dataset
% https://www.mathworks.com/help/stats/fitnlm.html#d122e353492
Y = e*2;
X = 1:length (e);
figure;
plot(X, Y);
hold on;

modelfun = @(b,x)(b(1) + b(2)*sin(2*pi*x/b(3) + b(4)))

B0 = trimmean(Y,10)  % Vertical shift
B1 = (max(Y) - min(Y))/2; % Amplitude
B2 = 12; % Phase (Number of peaks)
B3 = 0; % Phase shift (eyeball the Curve)
beta0 = [B0 B1 B2 B3];

mdl = fitnlm(X,Y,modelfun,beta0)


plot (X,mdl.Fitted);
hold off;

%% Plotting 
figure;
plot( x, signal, 'c', x,y,'b',  X, mdl.Fitted, 'r' )
legend ('Original','Wavelet Denoised','Least square sinusoidal regression')
title ('Daily conversation duration')
xlabel ('Day')
ylabel ('Seconds')
grid
