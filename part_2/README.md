## Part 2: Clinical Application Overview

Now that you have built your pulse rate algorithm and tested your algorithm to know it works, we can use it to compute more clinically meaningful features and discover healthcare trends.

Specifically, you will use 24 hours of heart rate data from 1500 samples to try to validate the well known trend that average resting heart rate increases up until middle age and then decreases into old age. We'll also see if resting heart rates are higher for women than men. See the trend illustrated in this image:

![heart-rate-age-ref-chart](heart-rate-age-reference-chart.jpg)

Follow the steps in the `clinical_app_starter.ipynb` to reproduce this result!

### Dataset (CAST)

The data from this project comes from the [Cardiac Arrythmia Suppression Trial (CAST)](https://physionet.org/content/crisdb/1.0.0/), which was sponsored by the National Heart, Lung, and Blood Institute (NHLBI). CAST collected 24 hours of heart rate data from ECGs from people who have had a myocardial infarction (MI) within the past two years.<sup>1</sup> This data has been smoothed and resampled to more closely resemble PPG-derived pulse rate data from a wrist wearable.<sup>2</sup>

1. **CAST RR Interval Sub-Study Database Citation** - Stein PK, Domitrovich PP, Kleiger RE, Schechtman KB, Rottman JN. Clinical and demographic determinants of heart rate variability in patients post myocardial infarction: insights from the Cardiac Arrhythmia Suppression Trial (CAST). Clin Cardiol 23(3):187-94; 2000 (Mar)
2. **Physionet Citation** - Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals (2003). Circulation. 101(23):e215-e220.


### Clinical Conclusion


1. *For women, we see* higher resting heart rates for age group 35 to 69 years old

2. *For men, we see* relatively lower variation in heart rates throughout 35 to 79 years olds.

3. *In comparison to men, women's heart rate is* larger in variation.

4. *What are some possible reasons for what we see in our data?*

  - The biggest possibility for the large variance in women's data is the number of samples. The number of rows for women is 277 while for men is 1260. If there is not enough sample, very large or low values would skew the distribution, hence causing the large variance.
  - Another reason is the data not taken from a wide enough sample that they exhibit similar trends. This is probably more likely with the male data.
  - Obviously, we can't rule out the possibility of faulty apparatus. I would check the data for null values to find this out.

5. *What else can we do or go and find to figure out what is really happening? How would that improve the results?*

  - As mentioned above, checking the data for null values is a standard first step.
  - Since the data came from the Cardiac Arrythmia Suppression Trial (CAST) dataset, the sample is not representative for the entire population of males and females. How about the healthy ones, how about people with other conditions?

  Therefore, to improve the results, it is important to gather more data, both from healthy subjects and other conditions.

6. *Did we validate the trend that average resting heart rate increases up until middle age and then decreases into old age? How?*

  Due to the irrepresentative sample, I do not think it is possible to conclude that with this dataset.