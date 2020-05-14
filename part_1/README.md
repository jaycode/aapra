## Part 1: Pulse Rate Algorithm

### Project Description

We built an algorithm to predict heart rate, measured in beats per minute (BPM), using wrist-type photoplethysmographic (PPG) and Inertial Measurement Unit (IMU) signals that were recorded during various activities. We also simultaneously record each participant's ECG signals from their chest using wet ECG sensors. The recorded ECG signals were then converted into heart rates, calculated in beats per minute (BPM) and used as a reference to measure the algorithm's performance.

This problem is challenging especially during intensive exercises since the signals are contaminated by extremely strong motion artifacts (MA) caused by subjects' hand movements.

In the "Code" section below, a code cell can be run to execute the algorithm with a given dataset and measure its performance.

### Data Description

We will be using the **Troika**[1] dataset to build our algorithm. Find the dataset under `datasets/troika/training_data`. The `README` in that folder will tell you how to interpret the data. The starter code contains a function to help load these files.

1. Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. [Link](https://arxiv.org/pdf/1409.5181.pdf)

Some remarks on the number of channels of the signals:

- ECG signals have one channel.
- PPG signals have two channels. In this project, we take the second channel as it poses the more challenging problem.
- Accelerometers have three channels, each corresponding to a space axis (x, y, and z). We use the L2-norm of the magnitude of these three channels i.e. the distance measured by this formula: `sqrt(x**2 + y**2 + z**2)`.

#### Quantity of data 

The data were recorded from 12 subjects. 11 subjects were used for the training data and 1 subject for testing data. Each subject has at least 5-minutes worth of data of 125 Hz sample, so we are looking at `125 * 5 * 60 = 37500` rows of each signal per person, at least (there are a few seconds offsets for each recording).

### Algorithm Description

#### How the algorithm works

The algorithm works as follows:

1. Perform Fast Fourier Transformation to the PPG and IMU signals.
2. Find two strongest frequencies from the PPG data.
3. Find the peaks of PPG frequencies.
4. If no peak was found, simply get the strongest frequency and use it as the EEG estimation.
5. If the distance of magnitutes between these two frequencies is small (20% of the best signal's magnitude), perform the following algorithm to choose the best peak:
    - For each peak of the Fast-Fourier Transformed IMU spectrums, see how close it is is to each peak of the PPG spectrums.
    - Choose the closest peak's PPG frequency as the estimation.
6. If there is only a single peak, use the frequency of that peak as the EEG estimation.


Here are the steps to run the algorithm:

1. Run the code cell below to check the performance of the algorithm on the training data.
2. Testing data are available in the directory `datasets/troika/testing_data` if you'd like to see the performance of the model on new data.

The algorithm will then return two outputs: the estimated frequency (in BPM) and the confidence score of that prediction. The confidence score is calculated by dividing the sum of powers near the peak.

#### Specific aspects of the physiology that the algorithm takes advantage of

PPG signals can be used for measuring heart rate due to the following aspects of the physiology of human blood vessels: 1) capillaries in the wrist fill with blood when the ventricles contract; and 2) when the blood returns to the heart, there are fewer red blood cells in the wrist. During state 1, the (typically green) light emitted by the PPG sensor is absorbed by red blood cells in these capillaries and the photodetector will see the drop in reflected light. During state 2, fewer red blood cells in the wrist absorb the light and the photodetector sees an increase in reflected light. The period of this oscillating waveform is the pulse rate.

#### Factors used by the model

The model uses two factors: the PPG's frequency of the best-fitted peak and the L2-norm of Acceleration's frequency of the highest magnitude.

#### Description of the algorithm outputs

As mentioned above, the algorithm churns out two values: the predicted heart rate (BPM) and its confidence rate of the prediction. The higher the confidence rate is, (theoretically,) the more likely is the prediction to be correct.

#### Caveats on algorithm outputs

The confidence rate is only calculated based on the magnitude of a small area that contains the estimated spectral frequency relative to the sum magnitude of the entire spectrum.

#### Common failure modes

The condition that may cause the algorithm to fail is when the PPG picks a higher frequency signal that is not from the heart rate. This is possible due to MA contaminations from hand movements. To deal with this problem, we take into consideration the accelerations in the algorithm.

### Algorithm Performance

The performance was calculated by calculating the mean absolute error between the HR estimation and the reference HR from the ECG sensors at 90% availability. Put another way, 90% of the best estimates according to the algorithm's confidence scores. For this project, the requirement is to have lower than 15 BPM of this measurement error score on the test set.

The error of the model in the testing data is 13.63 BPM.

As we only have 12 subjects, the algorithm may not be as generalizable as we'd like when used for a large population. Additionally, there is also a limited variability in terms of activities and environments, which would render this model less usable to cycling data or to other labs or outside environments. 

The algorithm requires its parameters (especially in the `Predict()` function) to have the exact values that are currently used to achieve the desired low error score. To improve its generalizability, a machine learning algorithm may need to be employed, with also more test subjects performing various kinds of different activities.


### References

- Signal Processing basics: https://allsignalprocessing.com/introductory-content/
- Code for BandpassFilter: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
- Code for SSA: https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition
- Using `scipy.optimize`: https://stackoverflow.com/questions/51883058/l1-norm-instead-of-l2-norm-for-cost-function-in-regression-model/51896790

-----