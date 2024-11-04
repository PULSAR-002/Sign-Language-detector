I have used mediapipe to extract 21 landmarks for each sign with their corresponding labels and I have trained those using a machine learning algorithm called Random forest classifier.
To make your custom dataset follow the following steps but if you want to run the program with the existing data then skip collecting portion


1. run the data_collection.py
2. Enter the label (sign name) and press enter
3. the model will start collecting data when you hit space bar key and pause when you hit space bar key so it will be easy for you to collect the data
4. After collecting the data press 'q' so it will save that data for the corresponding label in a .csv file
5. If you want to collect data for another sign after pressing 'q' just type the name of the new label and repeat the process.


data_training
1. After you collected the data run the training file it will take a few minuites to train depending upon your data


detecting
1. After training run the real_time_detection.py 


my dataset
my dataset include data for 10 signs
hello==>
i like it==> 
i dont like it==> 
i love you==> 
yes==> fist
no==> 
wonderful==>
peace==>
thank you==>
eat==>
drink==>


caution: If you are using my collected dataset then you have to train the model again as I trained it but the trained file exceeds 25 MB so i am unable to upload it here <3<3
I hope all of your doubts are clear but if there are some then feel free to contact me via email abdullahqazi002@gmail
remember me in your prayers <3
