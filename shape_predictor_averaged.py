# -*- coding: utf-8 -*-
# THIS ONE IS THE ONE THAT WORKS
"""
Created on Fri Dec  6 16:58:12 2019

@author: monre
"""

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import pycpd.rigid_registration_no_scaling_testing as rigid_registration_no_scaling_testing
import pycpd.rigid_registration as rigid_registration
import numpy as np
import time
import open3d as o3d
import os
import shutil
from stl import mesh

def main(f_femur_target, f_model):
    # reg = rigid_registration(**{'X': femur_target, 'Y': model})
    reg = rigid_registration(**{'X': f_femur_target, 'Y': f_model})  # register the femur surface from the Depth Camera to
    # the model, minimising the distance between the points of the femur surface and the model
    reg.register()
    f_dictionary = {"transformed_model": reg.register()[0],
                    "probability": reg.register()[2],
                    "rotation_matrix": reg.register()[3],
                    "translation": reg.register()[4]}
    return f_dictionary


pca_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/OAI-ZIB/processed_data/07.pca_tests/deformable'
pca_mean = np.load(pca_path+'/pca_mean.npy')

vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 45]
femur_id = [9002430, 9287216, 9414083, 9581253, 9833489]
faces = 200

for j in range(len(femur_id)):
    total_iter = []
    total_time = []
    target_path = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/averaged/' \
                  + str(faces) + '_faces/femur_bone_' + str(faces) + '_faces_' + str(femur_id[j]) + '.ply'
    target_load = o3d.io.read_point_cloud(target_path)
    femur_target = np.asarray(target_load.points)

    error = []
    real_load = o3d.io.read_point_cloud('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/femur_bone_low_downsample_' + str(femur_id[j]) + '.ply')
    femur_real = np.asarray(real_load.points)
    for i in range(len(vals)):
        start_time = time.time()
        PCs = vals[i]
        #MODEL
        pca_components = np.load(pca_path+'/pca_components.npy')[0:PCs, :]
        X_new = np.matmul(np.zeros([1, PCs]), np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
        model = np.zeros([int(X_new.size / 3), 3])
        model[:, 0] = X_new[0, 0:int(X_new.size / 3)]
        model[:, 1] = X_new[0, int(X_new.size / 3):int(2 * X_new.size / 3)]
        model[:, 2] = X_new[0, int(2 * X_new.size / 3):int(X_new.size)]

        dictionary = main(femur_target, model)
        transformed_model = dictionary['transformed_model']
        probability = dictionary['probability']
        rotation_matrix = dictionary['rotation_matrix']
        translation = dictionary['translation']

        x0 = translation[0]
        y0 = translation[1]
        z0 = translation[2]
        b0 = np.arcsin(-rotation_matrix[2, 0])
        a0 = np.arcsin(rotation_matrix[2, 1]/np.cos(b0))
        g0 = np.arcsin(rotation_matrix[1, 0]/np.cos(b0))

        a = np.asarray([x0, y0, z0, a0, b0, g0])
        b = np.zeros([1, PCs])
        initial_guess = np.concatenate([a, b[0, :]])
        #UPDATE PARAMETERS
        '''
        os.chdir('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/DATA/OAI-ZIB/processed_data/08.testing')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_model)
        o3d.io.write_point_cloud('model.ply', pcd)
        pcd.points = o3d.utility.Vector3dVector(model_closest_points)
        o3d.io.write_point_cloud('model_closest_points.ply', pcd)
        '''
        #from sklearn.metrics import mean_squared_error
        #from math import sqrt

        def objective(params):
            #x, y, z, a, b, g, PC = params
            x = params[0]
            y = params[1]
            z = params[2]
            a = params[3]
            b = params[4]
            g = params[5]
            PC = params[6:]
            #MODEL DEFORMATION
            X_new = np.matmul(PC, np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
            model = np.zeros([int(len(X_new) / 3), 3])
            model[:, 0] = X_new[0:int(len(X_new) / 3)]
            model[:, 1] = X_new[int(len(X_new) / 3):int(2 * len(X_new) / 3)]
            model[:, 2] = X_new[int(2 * len(X_new) / 3):len(X_new)]
            #MODEL TRANSFORMATION
            t = np.asarray([x, y, z])
            T = np.tile(t, (model.shape[0], 1))
            R = np.asarray([[np.cos(g)*np.cos(b), -np.sin(g)*np.cos(a)+np.cos(g)*np.sin(b)*np.sin(a), np.sin(g)*np.sin(a)+np.cos(g)*np.sin(b)*np.cos(a)],
                            [np.sin(g)*np.cos(b), np.cos(g)*np.cos(a)+np.sin(g)*np.sin(b)*np.sin(a), -np.cos(g)*np.sin(a)+np.sin(g)*np.sin(b)*np.cos(a)],
                            [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
            transformed_model = np.dot(model, R) + T

            x1 = np.transpose(femur_target)  # converts femur_target into 3x54 - f1, f2, f3, f4, ... f54
            x2 = np.transpose(transformed_model)  # converts femur_target into 3x1995 - m1, m2, m3, m4, ... m1995
            x3 = np.repeat(x1, x2.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
            x4 = np.tile(x2, x1.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
            distances = np.linalg.norm(np.transpose(x3) - np.transpose(x4), axis=1).reshape(-1, 1995)  # finds normal distance for each
            # combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
            min_dist = np.min(distances, axis=1)  # find minimum distance for each femur point (f1, f2, f3, ... f54)
            error = np.power((np.power(np.sum(min_dist), 2) / femur_target.shape[0]), 0.5)  # finds rms error: sqrt(sum((fi-mi)^2)/n)
            # error = sqrt(np.sum(np.power(min_dist, 2)) / femur_target.shape[0])  # DOESN'T GIVE AS GOOD RESULTS AS THE TOP, at least for 3 PCs
            return error

        from scipy.optimize import minimize
        res = minimize(objective, initial_guess, tol=0.01)
        #print(res)
        final_params = res.x
        iterations = res.nit
        final_error = res.fun
        final_time = time.time() - start_time

        x = final_params[0]
        y = final_params[1]
        z = final_params[2]
        b = final_params[3]
        a = final_params[4]
        g = final_params[5]
        PC = final_params[6:]

        X_new = np.matmul(PC, np.transpose(np.linalg.pinv(pca_components))) + np.transpose(pca_mean)
        model = np.zeros([int(len(X_new) / 3), 3])
        model[:, 0] = X_new[0:int(len(X_new) / 3)]
        model[:, 1] = X_new[int(len(X_new) / 3):int(2 * len(X_new) / 3)]
        model[:, 2] = X_new[int(2 * len(X_new) / 3):len(X_new)]
        #MODEL TRANSFORMATION
        t = np.asarray([x, y, z])
        T = np.tile(t, (model.shape[0], 1))
        R = np.asarray([[np.cos(g)*np.cos(b), -np.sin(g)*np.cos(a)+np.cos(g)*np.sin(b)*np.sin(a), np.sin(g)*np.sin(a)+np.cos(g)*np.sin(b)*np.cos(a)],
                        [np.sin(g)*np.cos(b), np.cos(g)*np.cos(a)+np.sin(g)*np.sin(b)*np.sin(a), -np.cos(g)*np.sin(a)+np.sin(g)*np.sin(b)*np.cos(a)],
                        [-np.sin(b), np.cos(b)*np.sin(a), np.cos(b)*np.cos(a)]])
        transformed_model = np.dot(model, R) + T


        os.chdir('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/averaged/' + str(faces) + '_faces')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_model)
        o3d.io.write_point_cloud('femur_prediction' + str(PCs) + '_' + str(femur_id[j]) + '.ply', pcd)

        #np.save('params' + str(PCs) + '_' + str(femur_id[j]) + '.npy', final_params)
        np.save('iterations' + str(PCs) + '_' + str(femur_id[j]) + '.npy', iterations)
        np.save('error' + str(PCs) + '_' + str(femur_id[j]) + '.npy', final_error)
        np.save('time' + str(PCs) + '_' + str(femur_id[j]) + '.npy', final_time)

        total_iter.append(iterations)
        total_time.append(final_time)

        #RMSD
        prediction = transformed_model
        x1p = np.transpose(prediction)  # converts femur_target into 3x54 - f1, f2, f3, f4, ... f54
        x2p = np.transpose(femur_real)  # converts femur_target into 3x1995 - m1, m2, m3, m4, ... m1995
        x3p = np.repeat(x1p, x2p.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
        x4p = np.tile(x2p, x1p.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
        distancesp = np.linalg.norm(np.transpose(x3p) - np.transpose(x4p), axis=1).reshape(1995, -1)  # finds normal distance for each
        # combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
        min_dist1 = np.min(distancesp, axis=1)

        min_dist2 = np.min(distancesp, axis=0)
        # error.append((np.sum(min_dist)+np.sum(min_dist2)) / (prediction.shape[0]+femur_real.shape[0]))
        error.append(np.power((np.sum(np.power(min_dist1, 2)) + np.sum(np.power(min_dist2, 2))) / (prediction.shape[0] + femur_real.shape[0]), 0.5))  # RMS Error

    np.save('total_iter_' + str(femur_id[j]) + '.npy', np.array(total_iter))
    np.save('total_time_' + str(femur_id[j]) + '.npy', np.array(total_time))

    np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/averaged/' + str(faces) + '_faces/total_RMSD_error_' + str(femur_id[j]) + '.npy', np.array(error))

