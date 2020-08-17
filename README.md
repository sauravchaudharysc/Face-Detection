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



# OpenCV - HaarCascades

We are going to use one classifier which is already trained on a lot of face data. It is pretrained model. It basically looks for some special kind of features present on face. So we directly use this HaarCascades to detect faces. It uses **ADABOOST**

#### ADABOOST

We create base learner sequentially . Once it is trained and pass all the records and see how the model has performed. If any records have been incorrectly classified. So i will pass it to base learner 2 . And again if i get errors i will pass it to other base learner. This is the process how boosting techniques works.

AdaBoost is one of the first boosting algorithms to be adapted in solving practices. Adaboost helps you **combine multiple “weak classifiers” into a single “strong classifier”**.

[AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA)

## Haar-cascade Detection in OpenCV

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one.

Here we will deal with detection. OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in `haarcascade_frontalface_alt.xml` file. Let’s create face and eye detector with OpenCV.

First we need to load the required XML classifiers. 

```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). Once we get these locations, we can create a ROI(Region of Interest) for the face .

```
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
```

