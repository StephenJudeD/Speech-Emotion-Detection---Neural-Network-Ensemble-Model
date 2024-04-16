# Speech Emotion Detection - Neural Network Ensemble Model

## Introduction

This project explores the development of a neural network ensemble model for effective speech emotion detection. Speech emotion detection has important applications in areas such as human-computer interaction, mental health analysis, and customer service.

## Key Result - 1st Debate Biden & Trump - Final Segment

The key result is based on a video and transcript 2020 Joe Biden Donald Trump 1st Debate Part 2 (2020), hosted on rev.com and specifically the final segment. In order to fully explore this piece, the transcript has been added for context. This will be divided into two sections, firstly the transcript with both pieces, then visualisation by visualisation to compare both, and finally discuss the observations and compare. The segments were cut and converted using an editing tool, Figure 56 displays the two files used for this test. Both tested using the same parameters.
•	window_size = 1.0
•	hop_size = 0.2
•	confidence_threshold =0.7

**Trump Radar Chart**

![Trump Radar Chart](./images_ser/Trump_debate1.png)

---

**Biden Radar Chart**

![Biden Radar Chart](./images_ser/biden_debate_1.png)

Radar charts offer a valuable method for visually representing the emotional content conveyed during debates. They suggest that both Trump and Biden displayed negative emotions, but Trump's disgust was more noticeable. This finding aligns with previous studies on political communication, indicating that voters tend to be influenced more by negative emotions than positive ones.




## Process Flow

**Process Flow Diagram:**

![Process Flow Diagram](./images_ser/pres_data_flow_ensemble.jpg)

## Datasets

* Ravdess [Sample Audio File](./soundfiles/ravdess.wav)
* Ravdess Song [Sample Audio File](./soundfiles/rav_song.wav)
* SAVEE [Sample Audio File](./soundfiles/savee.wav)
* CREMA D [Sample Audio File](./soundfiles/CremaD.wav)
* TESS [Sample Audio File](./soundfiles/tess.wav)

*Disclaimer: Audio files cannot be directly played within the README.Please click the links to download and listen to the sample files.*

**All Datasets with Emotions**
![All Datasets](./images_ser/datasets_all.png)

**Final Balanced Dataset with 6 emtions**
![All Datasets](./images_ser/final_bar.png)


## Methodology

**Data Augmentation:**
* **add_echo()**: Adds an echo effect to the audio, simulating sound reflections.
* **add_background_noise()**: Mixes in realistic background noise from a provided audio file.
* **match_duration() / match_length()**: Makes audio clips a consistent length by repeating or trimming the audio.
* **add_reverb()**: Simulates reverberation, creating the sense of audio occurring within a space.
* **stretch()**: Changes the speed of the audio without affecting its pitch.
![Stretch](./images_ser/waveform_stretch.png)
* **shift()**: Applies a circular time shift to the audio.
* **pitch()**: Changes the pitch of the audio without affecting the speed.


**Feature Extraction:** 
* **MFCCs (Mel-Frequency Cepstral Coefficients):** Capture the shape of the vocal tract, commonly used in speech recognition and other audio tasks.
* **Delta MFCCs:** Measure the changes in MFCCs over time, providing dynamic information.
* **Acceleration MFCCs:**  Calculate the second-order derivative of MFCCs, emphasizing higher-order spectral changes.
* **Mel Spectrogram:** A visual representation of the audio's frequency content over time, useful for analyzing various audio patterns.
* **FRFT (Fractional Fourier Transform):** A generalization of the Fourier transform, it allows extracting features at intermediate points between the time and frequency domains.
* **Spectral Centroid:**  Indicates where the "center of mass" of the spectrum is located, related to the brightness of the sound.




**Preprocessing:**
* **Loading and Resampling:** Loads audio files and resamples them to a consistent sample rate. 
* **Silence Trimming:** Removes leading and trailing silence from the audio.
* **Normalization:** Adjusts audio amplitude to a standard range.

# Neural Network Ensemble Architecture

This project utilizes an ensemble of neural networks for enhanced performance and robustness. The ensemble includes the following models:

**Model 1: CNN, LSTM, GRU with Attention** 
* Employs a Convolutional Neural Network (CNN) for feature extraction.
* Includes Long-Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) layers to process sequential data.
* Incorporates an attention mechanism to focus on the most relevant features.


**Model 2: CNN Only**
* Features a streamlined CNN-based architecture for feature extraction.


**Model 3: CNN and LSTM**
* Combines CNN feature extraction with an LSTM layer for processing sequences and potentially capturing longer-term dependencies.


**Model 4: CNN and GRU**
* Employs CNN for feature extraction, followed by a Gated Recurrent Unit (GRU) layer for sequential processing (an alternative to the LSTM in Model 3).


**Model 5: CNN, Bidirectional LSTM, Bidirectional GRU with Multi-Head Attention**
* Leverages CNN for feature extraction.
* Utilizes bidirectional LSTM and GRU layers for capturing information from both past and future contexts.
* Employs multi-head attention to focus on various aspects of 



## Results

* Summarize performance metrics


* Discuss key findings from cross-corpora testing.

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
