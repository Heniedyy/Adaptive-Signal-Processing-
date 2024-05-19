Audio Signal Processing and Machine Learning
Project Overview

This project involves processing audio signals with noise reduction, visualization of the model, and machine learning modeling using Python. The project includes:

    noise.py: Contains functions for adding and filtering noise in audio signals.
    test.py: A script to test the functionality of the machine learning model.
    view.py: A script to visualize the model parameters and architecture.
    model.py: A script for building and training machine learning models on the processed audio signals.

Installation

    Python: Ensure you have Python installed on your system.
    Dependencies: Install the required dependencies using the following command:

    pip install -r requirements.txt

Usage
Noise Processing

To process audio signals with noise, use the functions defined in noise.py. This script includes functions such as lms_filter, nlms_filter, rls_filter, and hybrid_filter for different noise filtering algorithms and the mel spectrogram extraction.

Testing the Model

To test the machine learning model:

    Ensure your processed audio files are in the specified directory.
    Run the test.py script to test the trained model on the test dataset.


python test.py

Visualizing Model Parameters

To visualize the model parameters and architecture, run the view.py script. This script prints the details of the model's architecture and parameters.

python view.py

Machine Learning Model

To build and train machine learning models on the processed audio signals, run the model.py script. This script processes the audio signals, extracts features, and trains a model.


python model.py

Dependencies

    Python 3.x
    Libraries: Install the dependencies using pip install -r requirements.txt.

Contact

For any questions or issues, please contact me at youssef.mohamad@yahoo.com].