# Problem Statement:

Project tusk is to detect type of movement with respect to given RED PATAYA sensor, using ultrasonic sonar sensor signal. Sensor gives signal ADC data. Each signal consists of 50,017 data points. 50000 of which are signal data and rest are header data. I had to build a machine learning model to predict type of movement of human or other objects using Multi-layer Perceptron and Support Vector Machine algorithms.

# Collected Data:

I have collected data of three movement types, perpendicular or side to side movement, horizontal or toward and away movement and no movement. For perpendicular and no movement type of data I collected 14000 samples each at various distances. And for horizontal or toward and away movement I have around 9000 data samples. 

# File structure:

#### inference-
This folder consists of one python file that generates predictions on new unseen data.

#### Preprocessing-
This folder has three python files. One to load and view collected data. The signal_processing.py file as name suggests have algorithms used to process the signals. It had filtration, fft calculation, first peak index detection etc algorithms. The file feature_extraction.py file extracts statistical and frequency features from the processed signal.
*** IMPORTANT NOTE: DUE TO HARDWARE CONSTRAINTS I COULD NOT RUN THE PROGRAM SIMULTANEUOUSLY. SO RUNNING MAIN FUNCTION AT ONCE WOULD NOT WORK. 

#### Model training-
This folder has two python files for each type of model that was used. One for multi class svm-svc and another one is MLP.

#### Root folder-
joblib files are the saved pretrained models generated from the model training. csv files are various versions of datasets.

#### Result-
This folder contains generated Confusion matrix from the model. Also prediction score screenshots for both models are saved here.

# Environment Setup:
This whole project is using python 3.12. All the packages that were required in any parts of the project is added in the requirements.txt file. To setup the environment install python 3.12 on your system and follow the instruction below.

- Create virtual environment
```python -m venv /path/to/your/venv```

- Install from requirements.txt
```pip install -r /path/to/requirements.txt```

- Activate the virtual environment
Windows: ```YOUR-VENV\scripts\activate```
Linux: ```source YOUR-VENV/bin/activate```

***DONE***

# Executing the program:

- First execute the code file signal-processing. After running first line it would create filtered files in the same directory as raw files. You need to move the raw files in order to run it smoothly. Second line will create fft converted signal files from filtered signals. Now both this filtered and fft files should be moved into different locations. Next two functions will use the filtered signal files. So change the function passing parameters accordingly. 

- After that it is time to run the feature_extraction.py file. It will use the generated files before to calculate features. first two functions ```filtered_signal_feature_extraction``` and ```extract_frequency_features``` uses same location to read files from. You should swap or change location based on your system or preferences. Later the calculated feature files are combined to get one single feature file.

- Now its time to run the model.py files from 'modeltraining' folder. Each one will run seperately and generate predictions. SVM will create a Confusion Matrix and show Precision, Recall, f1 score, Accuracy score on terminal. MLP with create confusion matrix and show accuracy and loss function value on terminal. Both will save the trained model as .joblib file.

- The inference folder is to get prediction on new data from the saved pre-trained models. The preffared has to be uncommented from the main function. At first change the file location where the input data is and where do you want the prediction result to be. Than simply run the code. It will process the signal accordingly and generate features and than will make predictions,

# Future Improvements
The files cannot be run without editing file location explicitly. Which is frustating. In near future I will add edited files dictionary to be returned from the functions. So no need of changing the files location. It will automatically get the files that are needed. 

In far future, may also add docker and api to deploy it online.

