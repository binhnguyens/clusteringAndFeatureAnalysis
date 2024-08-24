function [output] = trend_extract(filename)
%     [amplitude,period,phase,intercept,output] = trend_extract(filename)

    n_subjects = 38;
    % amplitude = zeros (n_subjects);
    % period = zeros (n_subjects);
    % phase = zeros (n_subjects);
    % intercept = zeros (n_subjects);
    
    C = fileread (filename);
    
    match = ["[","]"];
    f = erase(C,match); % Erase
    f1 = strtrim(split (f)); % Remove \n
    
    for i = 1:(n_subjects)
        amplitude (i) = str2double(f1(i*4-3));
        period (i) = str2double(f1(i*4-2));
        phase (i) = str2double(f1(i*4-1));
        intercept (i) = str2double(f1(i*4));
    end
    
    output = zeros (n_subjects,4);
    output (:,1) = amplitude;
    output (:,2) = period;
    output (:,3) = phase;
    output (:,4) = intercept;
end