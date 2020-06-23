import numpy as np
import open3d as o3d
import os
import shutil

downsampled = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/VCG/MeshLab/03.downsampled_PCDs'
real_load = o3d.io.read_point_cloud(downsampled + '/femur_bone_downsampled_9002430.ply')

downsampled = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/'
real_load = o3d.io.read_point_cloud(downsampled + '/femur_bone_low_downsample_9002430.ply')

femur_real = np.asarray(real_load.points)


error = []
total_error = []
faces = 300
results = 'C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/' + str(faces) + '_faces_more'
for filename in os.listdir(results):
    if filename[0:5] == 'femur':
    #if filename == 'femur_prediction27.ply':
        a = filename
        prediction_load = o3d.io.read_point_cloud(results + '/' + filename)
        prediction = np.asarray(prediction_load.points)
        x1 = np.transpose(prediction)  # converts femur_target into 3x54 - f1, f2, f3, f4, ... f54
        x2 = np.transpose(femur_real)  # converts femur_target into 3x1995 - m1, m2, m3, m4, ... m1995
        x3 = np.repeat(x1, x2.shape[1], axis=1)  # creates f1, f1, f1, ... f2, f2, f2, ... f3, f3, f3, ...
        x4 = np.tile(x2, x1.shape[1])  # creates m1, m2, m3, m4, ... m1, m2, m3, m4, ... m1, m2, m3, m4, ...
        distances = np.linalg.norm(np.transpose(x3) - np.transpose(x4), axis=1).reshape(1995, -1)  # finds normal distance for each
        # combination of f and m, and reshapes the matrix into [[(f1-m1), (f1-m2), (f1-m3), ...], [(f2-m1), (f2-m2), (f2-m3), ...], [(f3-m1), (f3-m2), (f3-m3), ...], ...]
        min_dist = np.min(distances, axis=1)

        min_dist2 = np.min(distances, axis=0)
        #error.append((np.sum(min_dist)+np.sum(min_dist2)) / (prediction.shape[0]+femur_real.shape[0]))
        error.append(np.power((np.sum(np.power(min_dist, 2)) + np.sum(np.power(min_dist2, 2))) / (prediction.shape[0] + femur_real.shape[0]), 0.5)) #RMSD Error
        #error.append(np.max(min_dist)*np.max(min_dist2))

        #error.append(np.power(np.sum(np.power(min_dist2, 2)) / femur_real.shape[0], 0.5))

        #error.append(np.power((np.power(np.sum(min_dist), 2) / prediction.shape[0]), 0.5)) #The error used
        #error.append(np.power(np.sum(np.power(min_dist, 2)) / prediction.shape[0], 0.5))
        #error.append(np.sum(min_dist) / prediction.shape[0])
        #error.append(max(min_dist))

    #if filename[0:5] == 'error':
        #fun_error = np.load('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/300 points/' + filename)
        #total_error.append(fun_error)

np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/' + str(faces) + ' points_more/total_RMSD_error.npy', np.array(error))
#np.save('C:/Users/monre/OneDrive - Imperial College London/ME4/FYP/code/SSM/Results/Point Cloud/300 points/total_error.npy', np.array(total_error))
print('done')
