# Face Detection using HaarCascades

#### Pre-Requisite

1. Install Python
2. Install Python Library opencv
   -   pip install opencv-python

#### Run

1. Download the repository
2. Go to project folder
3. Open you terminal or cmd
4. Type python face_detection.py



OpenCV - HaarCascades

We are going to use one classifier which is already trained on a lot of face data. It is pretrained model. It basically looks for some special kind of features present on face. So we directly use this HaarCascades to detect faces. It uses **ADABOOST**

#### ADABOOST

We create base learner sequentially . Once it is trained and pass all the records and see how the model has performed. If any records have been incorrectly classified. So i will pass it to base learner 2 . And again if i get errors i will pass it to other base learner. This is the process how boosting techniques works.

AdaBoost is one of the first boosting algorithms to be adapted in solving practices. Adaboost helps you **combine multiple “weak classifiers” into a single “strong classifier”**.

[AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA)





