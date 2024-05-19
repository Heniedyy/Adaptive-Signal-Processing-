function NoiseFilter

    % Create the main figure
    hFig = figure('Toolbar', 'none', ...
                  'Menubar', 'none', ...
                  'Name', 'Noise Filter App', ...
                  'NumberTitle', 'off', ...
                  'Position', [100, 100, 800, 600]);

    % Add a button to load the original audio file
    uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Load Original Audio', ...
              'Position', [20, 560, 150, 30], 'Callback', @loadOriginalAudioCallback);

    % Add a button to load the noise audio file
    uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Load Noise Audio', ...
              'Position', [180, 560, 150, 30], 'Callback', @loadNoiseAudioCallback);

    % Add a button to save collected data
    uicontrol('Parent', hFig, 'Style', 'pushbutton', 'String', 'Save Data', ...
          'Position', [500, 560, 150, 30], 'Callback', @(~, ~) Utils.saveCollectedData());

    % Add popup menu for algorithm selection
    hPopup = uicontrol('Parent', hFig, 'Style', 'popup', ...
                       'String', {'Select Algorithm', 'LMS', 'NLMS', 'RLS', 'Hybrid'}, ...
                       'Position', [340, 560, 150, 30]);

    % Initialize axes for the plots
    hAxesOriginal = subplot(3, 1, 1, 'Parent', hFig);
    title(hAxesOriginal, 'Original Signal');
    xlabel(hAxesOriginal, 'Sample');
    ylabel(hAxesOriginal, 'Amplitude');

    hAxesNoisy = subplot(3, 1, 2, 'Parent', hFig);
    title(hAxesNoisy, 'Noisy Signal');
    xlabel(hAxesNoisy, 'Sample');
    ylabel(hAxesNoisy, 'Amplitude');

    hAxesFiltered = subplot(3, 1, 3, 'Parent', hFig);
    title(hAxesFiltered, 'Filtered Signal');
    xlabel(hAxesFiltered, 'Sample');
    ylabel(hAxesFiltered, 'Amplitude');

    % Initialize variables to hold audio data and sampling frequency
    global y yNoise Fs yNoisy;
    y = [];
    yNoise = [];
    Fs = 0;
    yNoisy = [];

    % Callback function for loading original audio
    function loadOriginalAudioCallback(~, ~)
        [fileName, pathName] = uigetfile({'*.wav', '*.mp3'}, 'Select the Original Audio File');
        if isequal(fileName,0) || isequal(pathName,0)
            disp('User canceled');
            return;
        end
        [y, Fs] = audioread(fullfile(pathName, fileName));
        
        plot(hAxesOriginal, y, 'LineWidth', 1.5);
        title(hAxesOriginal, 'Original Signal');
        xlabel(hAxesOriginal, 'Sample');
        ylabel(hAxesOriginal, 'Amplitude');
        grid on;
    end

    % Callback function for loading noise audio
    function loadNoiseAudioCallback(~, ~)
        [fileName, pathName] = uigetfile({'*.wav', '*.mp3'}, 'Select the Noise Audio File');
        if isequal(fileName,0) || isequal(pathName,0)
            disp('User canceled');
            return;
        end
        [yNoiseTmp, ~] = audioread(fullfile(pathName, fileName));

        % Ensure yNoise matches the length of y by padding or trimming
        if length(yNoiseTmp) > length(y)
            yNoise = yNoiseTmp(1:length(y));
        elseif length(yNoiseTmp) < length(y)
            yNoise = zeros(size(y));
            yNoise(1:length(yNoiseTmp)) = yNoiseTmp;
        else
            yNoise = yNoiseTmp;
        end

        yNoisy = y + yNoise;
        plot(hAxesNoisy, yNoisy, 'LineWidth', 1.5);
        title(hAxesNoisy, 'Noisy Signal');
        xlabel(hAxesNoisy, 'Sample');
        ylabel(hAxesNoisy, 'Amplitude');
        grid on;
        applySelectedFilter();
    end

% Function to apply selected filter and plot
function applySelectedFilter()
    if isempty(y) || isempty(yNoise)
        disp('Original and/or noise signal not loaded.');
        return; % Do not proceed if either signal is not loaded
    end

    algIdx = get(hPopup, 'Value');
    switch algIdx
         case 2 % LMS
            tic; % Start timer 
            [yFiltered, errorSignal] = AdaptiveFilters.filterLMS(yNoisy, 0.3);
            executionTime = toc; % Stop timer
           
        case 3 % NLMS
            tic; % Start timer timer
            [yFiltered, errorSignal] = AdaptiveFilters.filterNLMS(yNoisy, 0.3); 
            executionTime = toc; % Stop timer
           
        case 4 % RLS
            tic; % Start timer 
            [yFiltered, errorSignal] = AdaptiveFilters.filterRLS(yNoisy, 0.3); 
            executionTime = toc; % Stop timer
            
        case 5 % Hybrid
            tic; % Start timer 
            [yFiltered, errorSignal] = AdaptiveFilters.filterHybrid2(yNoisy, 0.3, 0.95, 0.5, 0.01); %stepSize, forgettingFactorRLS, mixParam, switchThreshold
            executionTime = toc; % Stop timer
           
        otherwise
            yFiltered = yNoisy; %no algorithm is selected
            
            errorSignal = []; % Set empty error signal for default case
            executionTime = 0; % Set execution time to 0 for default case
    end

    % Evaluate filters
    noisySignalPower = sum(yNoisy.^2);
    filteredSignalPower = sum((y - yFiltered).^2);
    snr = 10 * log10(noisySignalPower / filteredSignalPower);
    disp(['SNR for ', Utils.getAlgorithmName(algIdx), ': ', num2str(snr), ' dB']);

    mse = mean((y - yFiltered).^2);
    disp(['MSE for ', Utils.getAlgorithmName(algIdx), ': ', num2str(mse)]);

     % Convergence speed tracking
    if ~isempty(errorSignal)
        figure;
        plot(errorSignal);
        title(['Error Signal for ', Utils.getAlgorithmName(algIdx)]);
        xlabel('Sample');
        ylabel('Error');
    end

    % Computational complexity 
    disp(['Execution time for ', Utils.getAlgorithmName(algIdx), ': ', num2str(executionTime), ' seconds']);

    
    plot(hAxesFiltered, yFiltered, 'LineWidth', 1.5);
    title(hAxesFiltered, strcat('Filtered Signal - ', Utils.getAlgorithmName(algIdx)));
    xlabel(hAxesFiltered, 'Sample');
    ylabel(hAxesFiltered, 'Amplitude');
    grid on;
    
   
    end
end



   

    