# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 09:27:11 2019

@author: Admin
"""
import numpy as np
data = 'C:/Users/Admin/Documents/Centrale Paris/3A/OMA/NLP/Exo 3/convai2_fix_723.tar'

'''
import tarfile
tar = tarfile.open(data)
tar.extractall()
'''
rep = 'C:/Users/Admin/'
def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append( l.lower().split() )
    return sentences

train = text2sentences(rep+'train_none_original.txt')
true = text2sentences(rep+'train_none_original_no_cands.txt')

