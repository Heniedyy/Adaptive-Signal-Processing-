classdef FeatureExtraction
    methods (Static)
        % Compute MFCC features
        function mfccCoefficients = computeMFCC(signal, fs)
            frameLength = 0.025; % 25 milliseconds
    frameOverlap = 0.015; % 15 milliseconds
    numCoeffs = 13; % Number of MFCC coefficients

    % Define the window length
    windowLengthSamples = round(frameLength*fs);

    % Create a Hamming window vector
    windowVector = hamming(windowLengthSamples, 'periodic');

    coefficients = mfcc(signal, fs, ...
        'Window', windowVector, ...
        'OverlapLength', round(frameOverlap*fs), ...
        'NumCoeffs', numCoeffs);

    % Return the full matrix of MFCC coefficients
    mfccCoefficients = coefficients;
        end
function melSpectrogram = computeMelSpectrogram(signal, fs, nfft, numFilters)
            % Compute STFT
            windowLength = 0.025; % 25 milliseconds
            hopLength = 0.015; % 10 milliseconds
            window = hamming(round(windowLength * fs), 'periodic');
            overlap = round(hopLength * fs);
            [~, freqVector, timeVector, stftMatrix] = spectrogram(signal, window, overlap, nfft, fs);

            % Compute Mel filter bank
            melFilters = FeatureExtraction.melFilterBank(fs, nfft, numFilters);

            % Apply Mel filter bank to STFT
            melSpectrogram = melFilters * abs(stftMatrix);

            % Convert to dB scale
            melSpectrogram = 20 * log10(melSpectrogram + eps);

            % Transpose for correct orientation
            melSpectrogram = melSpectrogram';

            % Plot Mel spectrogram
            figure;
            imagesc(timeVector, freqVector, melSpectrogram);
            axis xy;
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            title('Mel Spectrogram');
            colorbar;
        end

        % Compute Mel filter bank
      function melFilters = melFilterBank(fs, nfft, numFilters)
    % Compute Mel scale range
    melMin = 0;
    melMax = 2595 * log10(1 + (fs / 2) / 700);

    % Compute Mel frequencies uniformly spaced in Mel scale
    melPoints = linspace(melMin, melMax, numFilters + 2); % Include both ends
    hzPoints = 700 * (10 .^ (melPoints / 2595) - 1); % Convert Mel to Hz scale

    % Convert Hz frequencies to FFT bin indices
    fftBins = floor((nfft + 1) * hzPoints / (fs / 2));

    % Create the Mel filter bank matrix
    melFilters = zeros(numFilters, nfft / 2 + 1);
    for i = 1:numFilters
        for k = fftBins(i):fftBins(i+1)
            if k >= 1 && k <= nfft / 2 + 1
                melFilters(i,k) = (k - fftBins(i)) / (fftBins(i+1) - fftBins(i));
            end
        end
        for k = fftBins(i+1):fftBins(i+2)
            if k >= 1 && k <= nfft / 2 + 1
                melFilters(i,k) = (fftBins(i+2) - k) / (fftBins(i+2) - fftBins(i+1));
            end
        end
    end
end

        % Compute signal features
        function featMat = computeSignalFeatures(signal, Fs, windowSize, hopSize)
            %% Initialize variables
    numWindows = floor((length(signal) - windowSize) / hopSize) + 1;
    featMat = zeros(numWindows, 7); % 7 features per window, for each window
     windowDuration_ms = (windowSize - 1) / Fs * 1000; % Convert from seconds to milliseconds
    fprintf('Window Duration: %.2f ms\n', windowDuration_ms);
    
    % Iterate over windows
    for i = 1:numWindows
        startIdx = (i - 1) * hopSize + 1;
        endIdx = startIdx + windowSize - 1;
        window = signal(startIdx:endIdx);

        % Time-domain features
        meanValue = mean(window);
        stdDev = std(window);
        rmsValue = sqrt(mean(window.^2));
        peakAmplitude = max(abs(window));
        energy = sum(window.^2);

        % Spectral centroid
        fftWindow = fft(window);
        magnitude = abs(fftWindow(1:floor(end/2)+1)); % Use only the positive frequencies
        totalPower = sum(magnitude);
        frequencyIndices = (0:length(magnitude)-1)'; % Column vector of indices
        weightedFrequency = sum(frequencyIndices .* magnitude); % Element-wise multiplication and sum
        spectralCentroid = weightedFrequency / totalPower;

        % Zero-crossing rate
        zeroCrossingRate = sum(abs(diff(window > 0))) / length(window);

        % Store features for the current window
        featMat(i, :) = [meanValue, stdDev, rmsValue, peakAmplitude, energy, spectralCentroid, zeroCrossingRate];
    end
        end

        % Plot the analysis window
        function plotAnalysisWindow(yFiltered, Fs, windowSize, hopSize)
            if isempty(yFiltered)
        disp('Filtered signal not available.');
        return;
    end

    % Calculate time in seconds
    time_s = (0:length(yFiltered)-1) / Fs;

    % Determine the number of windows
    numWindows = floor((length(yFiltered) - windowSize) / hopSize) + 1;

    % Plot the analysis window
    figure('Name', 'Analysis Window of Signal');
    hold on;
    plot(time_s, yFiltered, 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5); % Plot the original signal

    % Plot individual windows
    colors = lines(numWindows); % Generate distinct colors for each window
    for i = 1:numWindows
        startIdx = (i - 1) * hopSize + 1;
        endIdx = startIdx + windowSize - 1;
        windowData = yFiltered(startIdx:endIdx);
        windowTime = time_s(startIdx:endIdx);
        plot(windowTime, windowData, 'Color', colors(i, :), 'LineWidth', 1.5);
    end

    hold off;

    title('Analysis Window of Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');
    grid on;
        end
        
    end
    
    
end