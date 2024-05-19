Adaptive Filter Comparison and Feature Extraction
Project Overview

This project involves the implementation and comparison of various adaptive filtering algorithms for noise reduction in audio signals. The project includes:

    AdaptiveFilters.m: Implementation of adaptive filtering algorithms.
    AdaptiveFiltersT.m: An extension or variant of the adaptive filters.
    CompareFilters.m: Comparison of the performance of different adaptive filtering algorithms.
    ComputationTime.m: Measurement of computation times for each filtering algorithm.
    FeatureExtraction.m: Extraction of features from audio signals for further analysis or machine learning applications.
    NoiseFilter.m: A graphical user interface (GUI) for applying noise filters to audio signals.
    Utils.m: Utility functions used across various scripts.

Installation

   1) MATLAB: Ensure you have MATLAB installed on your system.
   2)Audio Files: Place your audio files in the appropriate directory as referenced in the scripts.


1)Adaptive Filtering

To apply the adaptive filters, use the functions defined in AdaptiveFilters.m and AdaptiveFiltersT.m. Example usage is provided in CompareFilters.m (AdaptiveFiltersT.m can have different tap orders).

2)Comparing Filters

To compare the performance of different filters:

    1)Ensure the audio files jarvus.wav and whiteNoise.wav are in the specified directory.
    2)Run the CompareFilters.m script to see the convergence speed of different filters.



3)Computation Time

To measure the computation time of each algorithm, run the ComputationTime.m script.

    1)Ensure the audio files jarvus.wav and whiteNoise.wav are in the specified directory.
    2)Run the CompareFilters.m script to see the convergence speed of different filters.

4)Noise Filtering GUI

To use the graphical user interface for noise filtering, run the NoiseFilter.m script. This GUI allows you to load original and noise audio files, select an algorithm, and visualize the filtering results.

Dependencies

    MATLAB with Signal Processing Toolbox
    Audio files: jarvus.wav and whiteNoise.wav

Contact

For any questions or issues, please contact me at youssef.mohamad@yahoo.com].
