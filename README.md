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
add_echo

Purpose: Simulates the effect of sound reflecting off surfaces, creating a sense of space.
How it Works:
Creates a delayed copy of the audio signal.
The delayed copy gets attenuated (reduced in volume) to simulate decay.
The delayed and attenuated signal is mixed back with the original.
add_background_noise

Purpose: Adds realistic environmental noise to make models more robust to real-world scenarios.
How it Works:
Loads a background noise audio file.
Adjusts the background noise amplitude to control its intensity.
Mixes the background noise into the original audio.
match_duration / match_length

Purpose: Ensures audio clips have a consistent length, which is often a requirement for audio analysis and machine learning models.
How it Works:
If the audio is shorter than the target length, it's repeated.
If the audio is longer, it's trimmed.
add_reverb

Purpose: Creates the impression of sound being produced in a room or other acoustic environment by simulating reflections and reverberation.
How it Works:
A delayed copy of the signal is created.
The delayed copy is decayed (reduced in amplitude)
The decayed, delayed signal is combined with the original.
stretch

Purpose: Changes the speed/tempo of the audio without affecting the pitch.
How It Works: Uses the librosa.effects.time_stretch function to change duration while preserving pitch.
shift

Purpose: Applies a circular shift to the audio data, effectively moving the sound's starting and ending points.
How It Works: Uses np.roll to shift the data within its array.
pitch

Purpose: Changes the pitch (highness or lowness) of the audio without affecting the tempo.
How It Works: Uses the librosa.effects.pitch_shift function to shift the pitch by a specified number of semitones.

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
