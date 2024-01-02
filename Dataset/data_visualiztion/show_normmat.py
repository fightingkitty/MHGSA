# -*- coding:utf-8 -*-
"""
@Time: 2023/1/14 23:01
@Author: Shuting Liu & Baochang Zhang
@IDE: PyCharm
@File: show_normmat.py
@Comment: #Enter some comments at here
"""
import scipy.io as sio

mat = sio.loadmat('/home/shuting/LProjects/Stroke_Radiomics/Dataset/Data/strokenormdataset/norm_mat.mat')
print(mat)