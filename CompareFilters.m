    function CompareFilters
        % Add the path for the filter functions
        addpath('C:\Users\youss\Matlab\Bachelor');
        % Load the audio files
        [y, Fs] = audioread('C:\Users\youss\Matlab\Bachelor\jarvus.wav'); 
        [yNoise, ~] = audioread('C:\Users\youss\Matlab\Bachelor\whiteNoise.wav'); 
        yNoise = yNoise(1:length(y));
        yNoisy = y + yNoise;
    
        % Apply Filter
        [~, errorSignalLMS] = AdaptiveFilters.filterLMS(yNoisy, 0.3);
        [~, errorSignalNLMS] = AdaptiveFilters.filterNLMS(yNoisy, 0.3);
        [~, errorSignalRLS] = AdaptiveFilters.filterRLS(yNoisy, 0.3);
        [~, errorSignalHybrid] = AdaptiveFilters.filterHybrid2(yNoisy, 0.3, 0.95, 0.5, 0.01);
    
        % Convert error signals to MSE and then to dB
        mseLMS = 10 * log10(errorSignalLMS.^2);
        mseNLMS = 10 * log10(errorSignalNLMS.^2);
        mseRLS = 10 * log10(errorSignalRLS.^2);
        mseHybrid = 10 * log10(errorSignalHybrid.^2);
    
        % Moving average for smoothing (window size 50)
        avgWindow = 50;
        avgMSELMS = movingAvg(mseLMS, avgWindow);
        avgMSENLMS = movingAvg(mseNLMS, avgWindow);
        avgMSERLS = movingAvg(mseRLS, avgWindow);
        avgMSEHybrid = movingAvg(mseHybrid, avgWindow);
    
        % Plotting
        figure;
        hold on;
        plot(avgMSELMS, 'g-', 'LineWidth', 1.5);
        plot(avgMSENLMS, 'b-', 'LineWidth', 1.5);
        plot(avgMSERLS, 'r-', 'LineWidth', 1.5);
        plot(avgMSEHybrid, 'k-', 'LineWidth', 1.5);
        legend('LMS', 'NLMS', 'RLS', 'Hybrid', 'Location', 'SouthEast', 'FontSize', 12);
        title('Speed of convergence (MSE in dB)');
        xlabel('Number of Iterations');
        ylabel('MSE (dB)');
        set(gca, 'XScale', 'log'); % Logarithmic x-axis scale
        grid on;
        hold off;
    end
    
    function avgData = movingAvg(data, windowSize)
        n = length(data);
        avgData = zeros(1, n);
        for i = 1:n
            if i < windowSize
                avgData(i) = mean(data(1:i));
            else
                avgData(i) = mean(data((i-windowSize+1):i));
            end
        end
    end