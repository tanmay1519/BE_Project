from matplotlib import test
from rest_framework.views import APIView
# from .models import *
from rest_framework.response import Response
from PIL import Image
import pickle
from tensorflow import keras
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU

allmodel = keras.models.load_model('my_model.h5',compile=False)
plantclassifierModel= keras.models.load_model('plants_classifier.h5',compile=False)
tomatocnn = keras.models.load_model('TomatoCNN.h5',compile=False)
potatocnn = keras.models.load_model('potatoCNN.h5',compile=False)
bellpeppercnn = keras.models.load_model('bellpepperCNN.h5',compile=False)
import numpy as np

# rf_tomato= pickle.load(open('model.pkl', 'rb'))
# rf_potato = pickle.load(open('potatomodel.pkl', 'rb'))
# rf_bell = pickle.load(open('bellmodel.pkl', 'rb'))

cnn_diseases=['bbs','b','peb','plb','p','tbs','teb','tlb','t']

plants = ['Bellpepper', 'Tomato', 'potato']
tomato_diseasescnn = ['tbs','teb','tlb','t']
potato_diseasescnn=['peb','plb','p']
bell_diseasescnn=['bbs','b']


tomato_diseasesrf = ['t','tbs','teb','tlb']
potato_diseasesrf=['p','p','peb','plb']
bell_diseasesrf=['b','bbs','b','b']

class GetTomato (APIView):
    def post (self,request) :
        output = {}

        cnnimg = request.data['CNNImg']
        print(len(cnnimg),len(cnnimg[0]))
        plant_name= plants[np.argmax(plantclassifierModel.predict([cnnimg])[0])]
        all_c_output= cnn_diseases[np.argmax(allmodel.predict([cnnimg])[0])]
        output["plantname"]=plant_name

        
        if plant_name == plants[0] :
            disease = bell_diseasescnn[np.argmax(bellpeppercnn.predict([cnnimg])[0])]
        elif plant_name == plants[1]:
            disease = tomato_diseasescnn[np.argmax(tomatocnn.predict([cnnimg])[0])]
        else :
            disease = potato_diseasescnn[np.argmax(potatocnn.predict([cnnimg])[0])]
        print(disease,all_c_output)
        if len (disease) == 1 :
                output["disease"]="Healthy"
        else :
                if disease[1]=="e":
                    output["disease"]="Early blight"
                elif disease[1]=="l":
                    output["disease"]="Late blight"
                else:
                    output["disease"]="Bacterial spot"
        if disease == all_c_output :
            output["confidence"] = "High"
            if len(disease) == 1 :
                output["comments"]="Take  regular care "
            else :
                 output["comments"]="Consult the expert"
      
        else :
            output["confidence"] = "Low"
            if len(disease) == 1 :
                output["comments"]="Needs Precuations to be taken"
            else :
                 output["comments"]="Consult the expert"

        return Response(output)

