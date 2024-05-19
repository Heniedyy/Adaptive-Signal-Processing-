classdef AdaptiveFilters
    methods (Static)
        
   % LMS filter
function [yFiltered, errorSignal] = filterLMS(yNoisy, mu)
    N = length(yNoisy);
    yFiltered = zeros(N, 1);
    w = 0.1; % Initial filter weight
    errorSignal = zeros(N, 1); % Initialize error signal
    for n = 2:N
        x = yNoisy(n-1); % Input for the filter
        d = yNoisy(n); % Desired output
        y = w * x; % Filter output
        e = d - y; % Error signal
        errorSignal(n) = e; % Store error signal
        w = w + mu * e * x; % Update filter weights
        yFiltered(n) = w * x; % Use updated filter to get filtered output
    end
end

% NLMS filter
function [yFiltered, errorSignal] = filterNLMS(yNoisy, mu)
    N = length(yNoisy);
    yFiltered = zeros(N, 1);
    w = 0; % Initial filter weight
    epsilon = 1e-6; % Small constant to avoid division by zero
    errorSignal = zeros(N, 1); % Initialize error signal
    for n = 2:N
        normFactor = norm(yNoisy(n-1))^2 + epsilon; % Norm of input signal
        yHat = w * yNoisy(n-1);
        e = yNoisy(n) - yHat;
        errorSignal(n) = e; % Store error signal
        w = w + (mu / normFactor) * e * yNoisy(n-1);
        yFiltered(n) = w * yNoisy(n-1); % Apply filter to get filtered output
    end
end
%RLS filter
function [yFiltered, errorSignal] = filterRLS(yNoisy, lambda)
    N = length(yNoisy);
    yFiltered = zeros(N, 1);
    w = 0; % Initial filter weight
    P = 1; % Initial inverse covariance matrix (for a 1-tap filter, this is a scalar)
    errorSignal = zeros(N, 1); % Initialize error signal properly

    for n = 2:N
        % Compute the Kalman gain
        k = P * yNoisy(n-1) / (lambda + yNoisy(n-1)' * P * yNoisy(n-1));
        % Compute the estimation error
        e = yNoisy(n) - w * yNoisy(n-1);
        % Store the error
        errorSignal(n) = e;
        % Update the filter weight
        w = w + k * e;
        % Update the inverse covariance matrix
        P = (1 / lambda) * (P - k * yNoisy(n-1)' * P);
        % Compute the filtered output
        yFiltered(n) = w * yNoisy(n-1);
    end
end


% Hybrid filter
function [yFiltered, errorSignal] = filterHybrid2(yNoisy, stepSizeNLMS, forgettingFactorRLS, mixParam, switchThreshold)
    N = length(yNoisy);
    yFiltered = zeros(N, 1); % Initialize output signal
    errorSignal = zeros(N, 1); % Initialize error signal array
    w = 0; % Start with a scalar for simplicity, representing filter coefficients

    for n = 2:N
        x = yNoisy(n-1); % Tap input
        d = yNoisy(n);   % Desired output
        
        % Hybrid filtering logic: assuming simple condition to switch between RLS and NLMS
        if abs(w * x - d) > switchThreshold
            % RLS-like update
            w = w + forgettingFactorRLS * (d - w * x) * x; % Simplified RLS update
        else
            % NLMS-like update
            w = w + (stepSizeNLMS / (x^2 + 1e-6)) * (d - w * x) * x; % NLMS update
        end

        yFiltered(n) = w * x; % Apply filter to get filtered output
        errorSignal(n) = d - yFiltered(n); % Calculate error signal
    end
end

    end
end




