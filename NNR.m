function NNR
    % Define the path to your files
    speechPath = 'C:\Users\youss\Matlab\Bachelor\speech.wav';
    noisePath = 'C:\Users\youss\Matlab\Bachelor\whiteNoise.wav';

    % Load the speech and noise files
    [speechSignal, Fs] = audioread(speechPath);
    [noiseSignal, ~] = audioread(noisePath);
    
    % Make sure the noise is the same length as the speech signal
    noiseSignal = noiseSignal(1:length(speechSignal));

    % Define SNR levels for simulation
    snrLevels = -10:1:10;  % From -10 dB to 10 dB SNR
    nrrValues = zeros(length(snrLevels), 4); % Store NRR for each filter

    % Loop over SNR levels to add noise and filter
    for i = 1:length(snrLevels)
        snr = snrLevels(i);

        % Create noisy signal at current SNR
        noisySignal = addNoise(speechSignal, noiseSignal, snr);

        % Apply filters
        [~, errorLMS] = AdaptiveFilters.filterLMS(noisySignal, 0.03);
        [~, errorNLMS] = AdaptiveFilters.filterNLMS(noisySignal, 0.03);
        [~, errorRLS] = AdaptiveFilters.filterRLS(noisySignal, 0.03);
        [~, errorHybrid] = AdaptiveFilters.filterHybrid2(noisySignal, 0.03, 0.95, 0.5, 0.01);

        % Compute NRR for each filter
        nrrValues(i, 1) = calculateNRR(speechSignal, errorLMS);
        nrrValues(i, 2) = calculateNRR(speechSignal, errorNLMS);
        nrrValues(i, 3) = calculateNRR(speechSignal, errorRLS);
        nrrValues(i, 4) = calculateNRR(speechSignal, errorHybrid);
    end

    % Plot NRR vs SNR for each filter
    figure;
    plot(snrLevels, nrrValues(:, 1), 'b-', 'LineWidth', 2);
    hold on;
    plot(snrLevels, nrrValues(:, 2), 'r--', 'LineWidth', 2);
    plot(snrLevels, nrrValues(:, 3), 'g-.', 'LineWidth', 2);
    plot(snrLevels, nrrValues(:, 4), 'k:', 'LineWidth', 2);
    xlabel('Input SNR (dB)');
    ylabel('NRR (dB)');
    legend('LMS', 'NLMS', 'RLS', 'Hybrid');
    title('SNR vs NRR of Adaptive Filters');
    grid on;
    hold off;
end

function noisySignal = addNoise(cleanSignal, noiseSignal, snr)
    % Calculate the power of the clean signal and noise
    signalPower = rms(cleanSignal)^2;
    noisePower = rms(noiseSignal)^2;

    % Scale noise to achieve desired SNR
    noiseScale = sqrt(signalPower / (noisePower * 10^(snr/10)));
    scaledNoise = noiseSignal * noiseScale;

    % Add scaled noise to clean signal
    noisySignal = cleanSignal + scaledNoise;
end

function nrr = calculateNRR(cleanSignal, errorSignal)
    % Calculate initial and error powers
    initialNoisePower = var(cleanSignal - mean(cleanSignal));
    errorPower = var(errorSignal - mean(errorSignal));

    % Ensure errorPower is not zero
    if errorPower == 0
        errorPower = eps;  % Use epsilon to avoid division by zero
    end

    % Calculate NRR in dB
    nrr = 10 * log10(initialNoisePower / errorPower);
end

