function ComputationTime
    % Sample Signal
    [y, Fs] = audioread('C:\Users\youss\Matlab\Bachelor\jarvus.wav'); 
    [yNoise, ~] = audioread('C:\Users\youss\Matlab\Bachelor\whiteNoise.wav'); 
    yNoise = yNoise(1:length(y));
    yNoisy = y + yNoise;

    % Measure Computation Time
    timeLMS = timeFilter(@AdaptiveFiltersT.filterLMS, yNoisy, 0.3);
    timeNLMS = timeFilter(@AdaptiveFiltersT.filterNLMS, yNoisy, 0.3);
    timeRLS = timeFilter(@AdaptiveFiltersT.filterRLS, yNoisy, 0.3);
    timeHybrid = timeFilter(@AdaptiveFiltersT.filterHybrid2, yNoisy, 0.3, 0.95, 0.5, 0.01);

    % Data for plotting
    algorithms = {'LMS', 'NLMS', 'RLS', 'Hybrid'};
    times = [timeLMS, timeNLMS, timeRLS, timeHybrid];

    % Plotting
    figure;
    bar(times);
    set(gca, 'XTickLabel', algorithms);
    ylabel('Computation Time (seconds)');
    title('Computation Time for Different Adaptive Algorithms with Filter Order M = 100');
end

function timeTaken = timeFilter(filterFunc, yNoisy, varargin)
    tic; % Start timer
    filterFunc(yNoisy, varargin{:});
    timeTaken = toc; % Stop timer and record time
end
