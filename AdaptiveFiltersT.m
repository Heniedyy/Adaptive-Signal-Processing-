classdef AdaptiveFiltersT
   methods (Static)

       function [yFiltered, errorSignal] = filterLMS(yNoisy, mu)
    N = length(yNoisy);
    M = 10; % Filter order
    yFiltered = zeros(N, 1);
    w = zeros(M, 1); % Initialize filter weights as a vector
    errorSignal = zeros(N, 1);

    for n = M+1:N
        x = yNoisy(n-M:n-1); 
        d = yNoisy(n); % Desired output
        y = w' * x; % Filter output
        e = d - y; % Error signal
        errorSignal(n) = e; % Store error signal
        w = w + mu * e * x; % Update filter weights
        yFiltered(n) = w' * x; % Use updated filter to get filtered output
    end
       end
       function [yFiltered, errorSignal] = filterNLMS(yNoisy, mu)
    N = length(yNoisy);
    M = 10; % Filter order
    yFiltered = zeros(N, 1);
    w = zeros(M, 1); % Initialize filter weights
    epsilon = 1e-6; % Small constant to avoid division by zero
    errorSignal = zeros(N, 1);

    for n = M+1:N
        x = yNoisy(n-M:n-1); % Input vector
        normFactor = norm(x)^2 + epsilon; % Norm of input vector
        yHat = w' * x;
        e = yNoisy(n) - yHat;
        errorSignal(n) = e; % Store error signal
        w = w + (mu / normFactor) * e * x;
        yFiltered(n) = w' * x; % Apply filter to get filtered output
    end
end
function [yFiltered, errorSignal] = filterRLS(yNoisy, lambda)
    N = length(yNoisy);
    M = 10; % Filter order
    yFiltered = zeros(N, 1);
    w = zeros(M, 1); % Initial filter weight
    P = eye(M) * 1000; % Initialize inverse covariance matrix
    errorSignal = zeros(N, 1);

    for n = M+1:N
        x = yNoisy(n-M:n-1); % Input vector
        k = P * x / (lambda + x' * P * x);
        e = yNoisy(n) - w' * x;
        errorSignal(n) = e;
        w = w + k * e;
        P = (1 / lambda) * (P - k * x' * P);
        yFiltered(n) = w' * x;
    end
end
function [yFiltered, errorSignal] = filterHybrid2(yNoisy, stepSizeNLMS, forgettingFactorRLS, mixParam, switchThreshold)
    N = length(yNoisy);
    M = 10; % Filter order
    yFiltered = zeros(N, 1); % Initialize output signal
    errorSignal = zeros(N, 1); % Initialize error signal array
    w = zeros(M, 1); % Start with a vector for filter coefficients

    for n = M+1:N
        x = yNoisy(n-M:n-1); % Vector of M previous samples
        d = yNoisy(n); % Desired output
        
        % Calculate the filter output and error
        yHat = w' * x;
        e = d - yHat;

        % Hybrid filtering logic
        if abs(e) > switchThreshold
            % RLS-like update with mixing
            rlsUpdate = forgettingFactorRLS * e * x;
            nlmsNorm = x' * x + 1e-6; % Avoid division by zero
            nlmsUpdate = (stepSizeNLMS / nlmsNorm) * e * x;
            w = w + mixParam * rlsUpdate + (1 - mixParam) * nlmsUpdate;
        else
            % NLMS update
            norm_x = x' * x + 1e-6; % Normalization factor to avoid division by zero
            w = w + (stepSizeNLMS / norm_x) * e * x;
        end

        yFiltered(n) = w' * x; % Apply filter to get filtered output
        errorSignal(n) = e; % Calculate and store error signal
    end
end



    end 
end