# liveness-detection-opencv
This project is to help with a facial recognition system under development. This system will help to get live facial images.


<h3 style="color: green; text-align: center"><b>Libraries</b></h3>

<ol>
    <li>pip install keras</li>
    <li>pip install tensorflow</li>
    <li>pip install numpy</li>
    <li>pip install imutils</li>
    <li>pip install matplotlib</li>
    <li>pip install opencv_python</li>
</ol>


<h3 style="color: green; text-align: center"><b>Commands</b></h3>

<ol>
    <li>[Command to train and save the model.]</li>
    >>> python train_liveness.py -d ./dataset/ -m ./model -l le.pickle -p testplot
    <li>[Command to run the liveness check.]</li>
    >>> python liveness_demo.py --model ./model -l le.pickle -d face_detector
    <li>[Command to gather images using the webcam.]</li>
    >>> python gather_examples.py -i 0 -d face_detector -o test_
    <li>[Command to predict the image being fake or real]</li>
    >>> python liveness_photo_demo.py -i test_/wer.jpeg
</ol>