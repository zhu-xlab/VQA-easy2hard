#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 13:01:36 2018

@author: sylvain
"""

import matplotlib
matplotlib.use('Agg')

from models import LR1_ZIN_model
import VQALoader
import VocabEncoder
import torchvision.transforms as T
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import cv2

import torchvision

from sampler import ImbalancedDatasetSampler
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
import datetime
from shutil import copyfile

import pickle
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("Test-LR1-ZINLR1.txt")
handler.setLevel(logging.INFO)
logger.addHandler(handler)

Dataset = "LR"


def train(model, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, Dataset='LR'):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset), num_workers=2)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,RSVQA.parameters()), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(reduce = False)#weight=weights)
        
    trainLoss = []
    valLoss = []
    if Dataset == 'HR':
        accPerQuestionType = {'area': [], 'presence': [], 'count': [], 'comp': []}
    else:
        accPerQuestionType = {'rural_urban': [], 'presence': [], 'count': [], 'comp': []}
    OA = []
    AA = []
    lbd = 65535
    K = 0.5
    bestAA = 0
    vector_V = Variable(torch.tensor(train_dataset.get_V()))
    Class_v = torch.tensor(train_dataset.get_Cv())
    for epoch in range(num_epochs):
        with torch.no_grad():
            RSVQA.eval()
            runningLoss = 0
            nb_classes = 94
            confusion_matrix = torch.zeros(nb_classes-1, nb_classes-1)

            if Dataset == 'HR':
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'area': 0}
            else:
                countQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
                rightAnswerByQuestionType = {'presence': 0, 'count': 0, 'comp': 0, 'rural_urban': 0}
            count_q = 0
            for i, data in enumerate(validate_loader, 0):
                if i % 1000 == 999:
                    print(i/len(validate_loader))
                _, question, answer, image, type_str, image_original = data
                question = Variable(question.long()).cuda()
                answer = Variable(answer.long()).cuda().resize_(question.shape[0])
                image = Variable(image.float()).cuda()
                if modeltype == 'MCB':
                    pred, att_map = RSVQA(image,question)
                    #pred = RSVQA(image,question)
                else:
                    pred = RSVQA(image,question)
                loss = torch.mean(criterion(pred, answer))
                runningLoss += loss.cpu().item() * question.shape[0]
                
                _, preds = torch.max(pred, 1)
                
                answer = answer.cpu().numpy()
                pred = np.argmax(pred.cpu().detach().numpy(), axis=1)
                
                for j in range(answer.shape[0]):
                    countQuestionType[type_str[j]] += 1
                    if answer[j] == pred[j]:
                        rightAnswerByQuestionType[type_str[j]] += 1

            valLoss.append(runningLoss / len(validate_dataset))
            with open("valLoss.pkl","wb") as f:
                pickle.dump(valLoss,f)
            print('epoch #%d val loss: %.3f' % (epoch, valLoss[epoch]))
        
            numQuestions = 0
            numRightQuestions = 0
            currentAA = 0
            for type_str in countQuestionType.keys():
                if countQuestionType[type_str] > 0:
                    accPerQuestionType[type_str].append(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
                numQuestions += countQuestionType[type_str]
                numRightQuestions += rightAnswerByQuestionType[type_str]
                currentAA += accPerQuestionType[type_str][epoch]
                #experiment.log_metric
                print("Accuracy " + type_str + ':') 
                logger.info("Accuracy " + type_str + ':')
                print(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
                logger.info(rightAnswerByQuestionType[type_str] * 1.0 / countQuestionType[type_str])
            OA.append(numRightQuestions *1.0 / numQuestions)
            AA.append(currentAA * 1.0 / 4)
            if AA[-1]>bestAA:
                print("AA: "+str(AA[-1]))
                logger.info("AA: "+str(AA[-1]))
                print("OA: "+str(OA[-1]))
                logger.info("OA: "+str(OA[-1]))
                #torch.save(RSVQA.state_dict(),"Best_RSVQA-ZIN-LR1-Full3.pth")
                
        RSVQA.train()
        runningLoss = 0
        epoch_min_loss = 65535
        epoch_max_loss = 0
        train_sample_count = torch.zeros(4)


if __name__ == '__main__':
    disable_log = True
    batch_size = 280
    num_epochs = 300
    #num_epochs = 35
    learning_rate = 0.00001
    ratio_images_to_use = 1
    modeltype = 'MCB'
    Dataset = 'LR'

    if Dataset == 'LR':
        data_path = '/home/zhenghang/lowresolution/'#'/raid/home/sylvain/RSVQA_USGS_data/'#'../AutomaticDB/'
        allquestionsJSON = os.path.join(data_path, 'questions.json')
        allanswersJSON = os.path.join(data_path, 'answers.json')
        questionsJSON = os.path.join(data_path, 'LR_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'LR_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'LR_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'LR_split_test_questions.json')
        answersvalJSON = os.path.join(data_path, 'LR_split_test_answers.json')
        imagesvalJSON = os.path.join(data_path, 'LR_split_test_images.json')
        images_path = os.path.join(data_path, 'Images_LR/')
    else:
        data_path = '/home/zhenghang/highresolution/RSVQA_HR/'
        allquestionsJSON = os.path.join(data_path, 'USGSquestions.json')
        allanswersJSON = os.path.join(data_path, 'USGSanswers.json')
        questionsJSON = os.path.join(data_path, 'USGS_split_train_questions.json')
        answersJSON = os.path.join(data_path, 'USGS_split_train_answers.json')
        imagesJSON = os.path.join(data_path, 'USGS_split_train_images.json')
        questionsvalJSON = os.path.join(data_path, 'USGS_split_test_questions.json')
        answersvalJSON = os.path.join(data_path, 'USGS_split_test_answers.json')
        imagesvalJSON = os.path.join(data_path, 'USGS_split_test_images.json')
        images_path = os.path.join(data_path, 'Data/')
    encoder_questions = VocabEncoder.VocabEncoder(allquestionsJSON, questions=True)
    if Dataset == "LR":
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = True)
    else:
        encoder_answers = VocabEncoder.VocabEncoder(allanswersJSON, questions=False, range_numbers = False)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),            
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
      ])
    
    if Dataset == 'LR':
        patch_size = 256
    else:
        patch_size = 256  
    train_dataset = VQALoader.VQALoader(images_path, imagesJSON, questionsJSON, answersJSON, encoder_questions, encoder_answers, train=True, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size, aug_file='train_augment.pkl')
    validate_dataset = VQALoader.VQALoader(images_path, imagesvalJSON, questionsvalJSON, answersvalJSON, encoder_questions, encoder_answers, train=False, ratio_images_to_use=ratio_images_to_use, transform=transform, patch_size = patch_size, aug_file='test_augment.pkl')
    
    
    if modeltype == 'MCBO':
        pass
    else:
        RSVQA = LR1_ZIN_model.VQAModel(encoder_questions.getVocab(), encoder_answers.getVocab(), input_size = patch_size).cuda()
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            RSVQA = torch.nn.DataParallel(RSVQA)
        #load pretrained models
        pretext_model = torch.load('Best_RSVQA-ZIN-LR1-Full3.pth')
        model_state_dict = RSVQA.state_dict()
        state_dict = {k: v for k, v in pretext_model.items() if k in model_state_dict.keys()}
        model_state_dict.update(state_dict)
        RSVQA.load_state_dict(model_state_dict)
    RSVQA = train(RSVQA, train_dataset, validate_dataset, batch_size, num_epochs, learning_rate, modeltype, Dataset)
    
    
