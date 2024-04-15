# Speech Emotion Detection - Neural Network Ensemble Model

## Introduction

This project explores the development of a neural network ensemble model for effective speech emotion detection. Speech emotion detection has important applications in areas such as human-computer interaction, mental health analysis, and customer service.

## Datasets

* Ravdess
* Ravdess Song
* SAVEE
* CREMA D
* TESS

## Methodology

* **Data Augmentation:**
* **add_echo()**: Adds an echo effect to the audio, simulating sound reflections.
* **add_background_noise()**: Mixes in realistic background noise from a provided audio file.
* **match_duration() / match_length()**: Makes audio clips a consistent length by repeating or trimming the audio.
* **add_reverb()**: Simulates reverberation, creating the sense of audio occurring within a space.
* **stretch()**: Changes the speed of the audio without affecting its pitch.
* **shift()**: Applies a circular time shift to the audio.
* **pitch()**: Changes the pitch of the audio without affecting the speed.


* **Feature Extraction:** (List the techniques used)
* **Preprocessing:** (Outline the steps involved)
* **Neural Network Ensemble Architecture:** (Describe the model architecture)
* **Ensemble Decision Making:** (Explain how decisions are combined)

## Results

* Summarize performance metrics


* Discuss key findings from cross-corpora testing.

## Process Flow

* **Process Flow Diagram:** ![Process Flow Diagram](https://github.com/StephenJudeD/Speech-Emotion-Detection---Neural-Network-Ensemble-Model/assets/105487389/aa26ac67-48d3-4ebf-aa69-dd009cfcf634)

## Code Structure

* **ensemble_full_final:** (Explain the purpose of this file)
* **Predictor_final:** (Describe the functionality of this file)

## How to Run

1. **Set up environment:** (List dependencies, Python version, etc.)
2. **Download datasets:** (If applicable, provide download instructions)
3. **Execute ensemble_full_final:** (Explain any arguments or parameters required)
4. **Run Predictor_final:** (Explain any arguments or parameters required)

## Future Work

* (List potential areas for improvement or expansion)

## Contact

* (Provide your contact information for questions or feedback) 
