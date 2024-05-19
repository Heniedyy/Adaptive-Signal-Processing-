classdef Utils
     properties (Access = private)
        features = [];
        labels = {};
        count = 0;
        windowCounts = [];
    end
    methods (Static)
        % Get algorithm name
        function name = getAlgorithmName(index)
    algorithms = {'LMS', 'NLMS', 'RLS', 'Hybrid'};
    if index == 1
        name = 'None';
    elseif index > 1 && index <= length(algorithms) + 1
        name = algorithms{index - 1};
    else
        name = 'Unknown';
    end
        end

         function collectFeaturesAndLabel(featVec, algIdx, numWindows)
            persistent features labels windowCounts count
            
            if isempty(features)
                features = [];
                labels = {};
                count = 0;
                windowCounts = [];
            end
            
            features = [features; featVec];
            labels = [labels; Utils.getAlgorithmName(algIdx)];
            windowCounts = [windowCounts; numWindows];
            count = count + 1;
        end
        % Save collected data
        function saveCollectedData()
            featuresData = Utils();
            
            choice = questdlg('Do you want to save the current data before continuing?', ...
                               'Save Data', ...
                               'Yes', 'No', 'Cancel', 'Yes');

            switch choice
                case 'Yes'
                    save('audio_features.mat', 'featuresData');
                    disp(['Data saved to audio_features.mat, save count: ', num2str(featuresData.count)]);
                case 'No'
                    disp('Data not saved.');
                case 'Cancel'
                    return; % Stop the function if the user chooses to cancel
            end
        end
    end
end