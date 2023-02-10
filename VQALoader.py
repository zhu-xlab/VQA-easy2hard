#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 18:15:28 2018

@author: sylvain
"""

import os.path
import json
import random
import pickle

import numpy as np
from skimage import io

from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2

RANDOM_SEED = 42
countQuestionType = {'presence': 17008, 'count': 17089, 'comp': 22554, 'rural_urban': 572}
	
DTPE = "LR"

class VQALoader(Dataset):
    def __init__(self, imgFolder, images_file, questions_file, answers_file, encoder_questions, encoder_answers, train=True, ratio_images_to_use = 1, transform=None, patch_size=512,aug_file=''):
        self.transform = transform
        self.encoder_questions = encoder_questions
        self.encoder_answers = encoder_answers
        self.train = train
        l1_questions_file = aug_file
        
        vocab = self.encoder_questions.words
        self.relationalWords = [vocab['top'], vocab['bottom'], vocab['right'], vocab['left']]
        if DTPE == "HR":
            C_weights = {'area': 4, 'presence': 1, 'count': 4, 'comp': 3}
            C_types = {'area': 1, 'presence': 2, 'count': 3, 'comp': 4} #572, 17008, 17089, 22554
        else:
            C_weights = {'rural_urban': 1, 'presence': 1, 'count': 4, 'comp': 3}
            C_types = {'rural_urban': 1, 'presence': 2, 'count': 3, 'comp': 4} #572, 17008, 17089, 22554
        
        with open(questions_file) as json_data:
            self.questionsJSON = json.load(json_data)
            
        with open(answers_file) as json_data:
            self.answersJSON = json.load(json_data)
            
        with open(images_file) as json_data:
            self.imagesJSON = json.load(json_data)

        #with open(l1_questions_file,'rb') as pickle_data:
        #    self.l1questionsJSON = pickle.load(pickle_data)
        
        images = [img['id'] for img in self.imagesJSON['images'] if img['active']]
        images = images[:int(len(images)*ratio_images_to_use)]
        self.images = np.empty((len(images), patch_size, patch_size, 3))
        
        self.len = 0
        for image in images:
            self.len += len(self.imagesJSON['images'][image]['questions_ids'])
        self.images_questions_answers = [[None] * 4] * self.len
        
        self.vector_V_init = np.ones([77232])
        self.class_V = np.zeros([77232])
        
        if DTPE=='HR':
            self.vector_V_init = np.ones([1125340])
            self.class_V = np.zeros([1125340])
        
        
        index = 0
        cnt = 0
        answers_t = set()
        self.train_labels = []
        for i, image in enumerate(images):
            #rnd_image = random.choice(images)
            img = io.imread(os.path.join(imgFolder, str(image)+'.tif'))
            #img = io.imread(os.path.join(imgFolder, str(rnd_image)+'.tif'))
            img = cv2.resize(img, (256,256))
            self.images[i, :, :, :] = img
            for questionid in self.imagesJSON['images'][image]['questions_ids']:
                question = self.questionsJSON['questions'][questionid]
            
                #question_str = 'How many ' + question["question"]
                question_str = question["question"]
                #l1_question_str = self.l1questionsJSON[questionid]
                #print(question_str,l1_question_str)
                aug_question_str = question_str
                #if len(l1_question_str[2])>0 and len(l1_question_str)>0:
                #    aug_question_str = random.choice(l1_question_str)
                #    aug_question_str = l1_question_str[2]
                #print(len(question_str))
                if random.random()>1:
                    aug_question_str = aug_question_str.replace('rural','country')
                    aug_question_str = aug_question_str.replace('urban','town')
                type_str = question["type"]
                answer_str = self.answersJSON['answers'][question["answers_ids"][0]]['answer']
                self.train_labels.append(self.encoder_answers.encode(answer_str)[0])
                
 
                if self.train:    
                    #self.vector_V_init[questionid] = C_weights[type_str]  #V0
                    self.vector_V_init[questionid] = C_weights[type_str] * float(len(question_str)) / 50. #V2
                    #self.vector_V_init[questionid] = float(len(question_str)) / 100. * C_weights[type_str] * 20000 / float(countQuestionType[type_str]) #v3
                    #print(type_str + ':' + str(self.vector_V_init[questionid]))
                    self.class_V[questionid] = C_types[type_str]
                    cnt += 1
                
                self.images_questions_answers[index] = [self.encoder_questions.encode(aug_question_str), self.encoder_questions.encode(question_str), self.encoder_answers.encode(answer_str), i, type_str, questionid]
                index += 1
        print(cnt)
    
    
    def get_V(self):
        min_V = np.min(self.vector_V_init)
        max_V = np.max(self.vector_V_init)
        #thresh_hold = min_V + (max_V - min_V) * 1 # V4
        thresh_hold = min_V + (max_V - min_V) * 0.6 # V2
        self.vector_V_init = (self.vector_V_init < thresh_hold).astype(np.float32)
        return self.vector_V_init
        
    def get_Cv(self):
        return self.class_V
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        question = self.images_questions_answers[idx]
        img = self.images[question[3],:,:,:]
        if self.train and not self.relationalWords[0] in question[1] and not self.relationalWords[1] in question[1] and not self.relationalWords[2] in question[1] and not self.relationalWords[3] in question[1]:
            if random.random() < .5:
                img = np.flip(img, axis = 0)
            if random.random() < .5:
                img = np.flip(img, axis = 1)
            if random.random() < .5:
                img = np.rot90(img, k=1)
            if random.random() < .5:
                img = np.rot90(img, k=3)
        if self.transform:
            imgT = self.transform(img.copy())
        if self.train:
            return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), np.array(question[2], dtype='int16'), imgT, question[4], question[5]
        else:
            return np.array(question[0], dtype='int16'), np.array(question[1], dtype='int16'), np.array(question[2], dtype='int16'), imgT, question[4], T.ToTensor()(img / 255)   
