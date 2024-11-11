import matplotlib.pyplot as plt
import numpy as np
import os


def get_data(attack_method="TECB", dataset="cifar10", defense_features=4, update_mode="top_only_[1]"):
    if attack_method == "TECB":
        if dataset == "cifar10":
            if defense_features == 4:
                if update_mode == "top_only_[1]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[2]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[3]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[4]":
                    main_task_acc = []
                    asr = []
                else:
                    raise ValueError
            elif defense_features == 8:
                if update_mode == "top_only_[1]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[2]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[3]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[4]":
                    main_task_acc = []
                    asr = []
                else:
                    raise ValueError
            elif defense_features == 12:
                if update_mode == "top_only_[1]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[2]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[3]":
                    main_task_acc = []
                    asr = []
                elif update_mode == "top_only_[4]":
                    main_task_acc = []
                    asr = []
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError
    elif attack_method == "BadVFL":
        if dataset == "cifar10":
            if defense_features == 4:
                if update_mode == "top_only_[1]":
                    main_task_acc = [83.68, 83.81, 83.63, 83.65, 83.52, 83.52, 83.45, 83.51, 83.18, 82.88, 82.60, 82.43, 81.80, 81.42, 80.74, 79.53, 78.28, 77.31, 75.52, 73.52, 71.75, 69.75, 67.96, 66.07, 63.62, 62.34, 60.16, 58.18, 56.79, 56.06, 53.93]
                    asr = [92.00, 93.30, 92.80, 92.90, 93.00, 93.50, 94.40, 94.20, 94.10, 94.40, 94.30, 94.50, 94.40, 94.50, 94.50, 94.50, 94.60, 94.20, 94.30, 94.10, 94.10, 93.40, 93.50, 93.10, 92.70, 92.60, 92.10, 90.80, 90.60, 90.00, 89.30]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [83.68, 83.69, 83.72, 83.72, 83.68, 83.73, 83.71, 83.67, 83.46, 83.31, 83.27, 83.02, 82.48, 82.29, 81.70, 80.07, 78.46, 77.01, 75.42, 73.73, 72.41, 70.93, 69.00, 66.85, 65.84, 64.59, 63.12, 61.74, 59.88, 58.65, 56.50]
                    asr = [92.00, 92.90, 92.70, 93.30, 93.10, 93.30, 93.40, 93.80, 94.20, 94.50, 94.50, 94.60, 95.30, 95.20, 95.20, 95.40, 95.50, 95.50, 95.30, 95.50, 95.60, 95.50, 95.50, 95.20, 95.50, 95.50, 95.30, 94.70, 94.20, 94.60, 93.70]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [83.68, 83.72, 83.69, 83.75, 83.76, 83.61, 83.55, 83.66, 83.53, 83.60, 83.56, 83.36, 83.23, 83.15, 83.07, 82.97, 82.70, 82.40, 82.19, 81.93, 81.36, 80.92, 80.49, 79.85, 78.99, 78.57, 76.99, 76.16, 74.42, 73.21, 72.53]
                    asr = [92.00, 92.60, 92.70, 92.60, 92.50, 92.40, 92.80, 92.60, 92.90, 92.60, 92.50, 92.60, 92.40, 93.00, 92.70, 92.80, 92.00, 92.70, 91.90, 91.80, 92.20, 91.90, 91.60, 90.00, 88.30, 87.30, 81.30, 71.90, 62.40, 57.20, 50.40]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [83.68, 83.68, 83.76, 83.67, 83.62, 83.78, 83.67, 83.64, 83.64, 83.70, 83.69, 83.79, 83.66, 83.67, 83.79, 83.60, 83.74, 83.72, 83.74, 83.62, 83.81, 83.85, 83.85, 83.82, 83.69, 83.75, 83.58, 83.64, 83.55, 83.73, 83.70]
                    asr = [92.00, 92.60, 92.90, 92.90, 92.60, 92.50, 92.60, 92.20, 92.50, 92.70, 92.50, 92.90, 92.60, 92.30, 92.70, 92.50, 93.30, 93.00, 92.70, 92.80, 92.60, 93.00, 92.70, 92.60, 93.10, 92.80, 93.00, 92.80, 93.00, 92.80, 92.60]
                else:
                    raise ValueError
            elif defense_features == 8:
                if update_mode == "top_only_[1]":
                    main_task_acc = [82.08, 81.82, 82.11, 82.00, 81.92, 81.80, 82.03, 82.03, 81.81, 81.62, 81.64, 81.56, 81.19, 80.80, 80.48, 80.21, 80.00, 79.49, 78.96, 78.41, 77.09, 76.77, 76.00, 75.26, 74.47, 73.70, 72.72, 71.24, 70.56, 69.41, 68.73]
                    asr = [94.00, 94.60, 94.30, 94.30, 94.10, 94.50, 93.80, 94.10, 93.70, 93.80, 93.40, 93.80, 93.10, 92.90, 92.90, 93.20, 92.60, 92.10, 91.80, 91.50, 90.90, 90.60, 89.70, 88.70, 88.90, 88.40, 87.70, 85.30, 85.40, 82.30, 81.10]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [82.08, 81.93, 81.84, 82.03, 82.04, 81.90, 82.11, 81.91, 81.77, 81.85, 81.51, 81.49, 81.14, 80.97, 80.69, 80.21, 80.08, 79.48, 79.33, 78.78, 78.05, 77.22, 76.24, 74.76, 73.35, 71.12, 69.21, 66.75, 65.06, 62.98, 61.46]
                    asr = [94.00, 94.50, 94.20, 94.60, 94.30, 94.10, 94.50, 94.30, 93.90, 93.20, 93.00, 92.70, 93.40, 93.00, 92.80, 91.60, 92.60, 91.90, 91.60, 90.50, 91.10, 90.00, 88.30, 86.40, 83.10, 81.80, 79.50, 75.10, 71.10, 68.10, 63.90]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [82.08, 82.00, 82.07, 82.01, 82.05, 82.04, 82.06, 81.93, 81.76, 81.98, 81.91, 81.78, 81.86, 81.67, 81.83, 81.73, 81.53, 81.43, 81.32, 81.36, 81.00, 80.89, 80.90, 80.77, 80.51, 80.33, 79.97, 79.84, 79.08, 79.17, 78.83]
                    asr = [94.00, 94.70, 94.50, 94.70, 94.00, 93.90, 94.40, 94.10, 94.40, 93.90, 94.10, 94.10, 93.90, 94.00, 94.00, 93.00, 93.50, 93.30, 93.30, 93.10, 93.00, 92.40, 92.40, 91.70, 90.70, 89.90, 89.00, 87.50, 86.80, 85.70, 86.30]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [82.08, 81.99, 81.96, 81.94, 82.03, 81.89, 82.08, 81.92, 81.95, 82.08, 81.96, 81.98, 82.01, 81.90, 82.02, 81.86, 82.02, 81.79, 81.98, 81.95, 81.81, 81.78, 81.79, 81.66, 81.77, 81.69, 81.73, 81.67, 81.53, 81.45, 81.50]
                    asr = [94.00, 94.90, 94.60, 94.80, 94.20, 94.50, 94.60, 94.90, 94.40, 94.40, 94.70, 94.90, 94.80, 94.80, 94.90, 94.90, 94.50, 94.50, 94.90, 94.40, 94.80, 94.90, 94.90, 94.90, 95.00, 94.90, 94.80, 94.60, 94.80, 95.10, 94.70]
                else:
                    raise ValueError
            elif defense_features == 12:
                if update_mode == "top_only_[1]":
                    main_task_acc = [80.62, 80.42, 80.28, 80.24, 80.16, 80.26, 80.06, 79.99, 79.80, 79.63, 79.23, 78.72, 78.29, 77.09, 75.78, 73.81, 71.32, 68.67, 66.45, 63.85, 61.43, 58.99, 56.34, 54.56, 52.68, 50.44, 49.03, 47.54, 45.75, 44.87, 43.30]
                    asr = [96.40, 97.30, 97.30, 97.10, 97.20, 97.20, 97.30, 96.90, 96.90, 96.80, 96.80, 96.90, 96.70, 96.70, 96.30, 96.40, 95.90, 94.80, 93.60, 93.80, 93.40, 92.40, 92.70, 91.60, 90.20, 88.70, 88.70, 87.90, 87.50, 86.80, 86.30]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [80.62, 80.23, 80.24, 80.31, 80.32, 80.19, 80.11, 79.85, 79.93, 80.00, 79.84, 79.36, 78.49, 77.89, 77.05, 75.77, 74.32, 72.37, 69.81, 67.32, 64.12, 61.64, 58.05, 54.20, 51.30, 47.95, 44.96, 42.60, 39.61, 37.73, 35.22]
                    asr = [96.40, 97.30, 97.20, 97.00, 96.90, 96.60, 96.60, 96.60, 96.50, 96.00, 96.00, 96.00, 95.70, 95.80, 95.60, 95.60, 95.60, 95.20, 95.10, 94.60, 94.80, 94.50, 94.40, 93.70, 93.50, 93.80, 93.00, 92.40, 91.30, 89.80, 89.20]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [80.62, 80.46, 80.36, 80.36, 80.30, 80.32, 80.38, 80.37, 80.28, 80.21, 80.02, 80.04, 79.69, 79.53, 78.96, 78.57, 77.81, 77.06, 76.33, 75.30, 74.17, 72.58, 70.94, 69.39, 67.83, 65.96, 63.49, 61.99, 59.15, 57.02, 54.69]
                    asr = [96.40, 97.00, 97.30, 97.00, 96.90, 97.00, 97.00, 96.80, 96.80, 96.50, 96.40, 96.70, 96.40, 96.50, 96.60, 96.60, 96.60, 96.40, 96.40, 96.60, 96.60, 96.60, 96.60, 96.60, 96.60, 96.40, 95.70, 96.40, 95.70, 95.90, 95.90]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [80.62, 80.49, 80.61, 80.28, 80.56, 80.47, 80.41, 80.39, 80.52, 80.41, 80.57, 80.56, 80.42, 80.21, 80.29, 80.45, 80.18, 80.35, 80.02, 80.25, 80.20, 80.04, 80.14, 80.03, 80.04, 79.98, 79.99, 79.85, 79.76, 79.78, 79.55]
                    asr = [96.40, 97.20, 97.20, 97.30, 97.10, 97.10, 97.20, 97.30, 97.30, 97.20, 97.10, 97.20, 97.30, 97.20, 97.30, 97.20, 97.40, 97.40, 97.40, 97.40, 97.60, 97.40, 97.60, 97.40, 97.50, 97.50, 97.50, 97.60, 97.50, 97.40, 97.50]
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError
    elif attack_method == "Villain":
        if dataset == "cifar10":
            if defense_features == 4:
                if update_mode == "top_only_[1]":
                    main_task_acc = [82.93, 83.04, 82.94, 82.43, 81.50, 80.35, 76.82, 71.44, 64.50, 58.11, 53.34, 48.63, 42.91, 39.80, 37.07, 34.97, 32.18, 28.28, 24.42, 22.16, 20.94, 18.25, 16.42, 15.20, 13.68, 12.58, 12.08, 12.28, 11.56, 11.08, 10.68]
                    asr = [99.74, 99.93, 99.87, 99.83, 99.76, 99.21, 95.40, 80.44, 56.71, 48.94, 42.36, 42.94, 62.20, 59.95, 62.66, 60.88, 41.46, 25.98, 3.06, 0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [82.93, 83.10, 82.96, 82.75, 81.20, 79.37, 76.08, 71.63, 64.38, 51.44, 41.72, 36.34, 30.56, 26.12, 23.52, 20.77, 19.69, 18.28, 16.57, 15.60, 14.79, 13.75, 13.19, 12.37, 11.65, 11.47, 10.83, 10.64, 11.15, 11.19, 10.82]
                    asr = [99.74, 99.63, 99.69, 99.97, 99.92, 99.46, 99.51, 99.70, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [82.93, 83.11, 82.69, 82.59, 82.01, 80.74, 79.21, 74.50, 70.80, 68.29, 65.82, 62.02, 57.58, 51.68, 43.56, 38.36, 34.63, 31.05, 28.44, 26.25, 22.87, 20.43, 18.97, 17.60, 16.86, 15.89, 15.68, 15.31, 14.76, 14.61, 14.35]
                    asr = [99.74, 99.31, 99.83, 99.22, 99.70, 99.78, 99.78, 99.95, 99.99, 99.96, 99.96, 99.99, 99.99, 100.00, 99.98, 99.96, 99.94, 99.86, 99.62, 99.31, 98.23, 96.17, 98.36, 96.11, 95.81, 93.33, 94.85, 95.41, 93.48, 95.75, 95.69]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [82.93, 83.03, 83.13, 83.06, 83.00, 83.03, 82.99, 82.91, 82.83, 82.72, 82.82, 82.74, 82.77, 82.60, 82.50, 82.65, 82.20, 82.34, 82.29, 82.16, 82.03, 82.04, 81.88, 81.71, 81.72, 81.66, 81.60, 81.30, 81.26, 81.31, 81.09]
                    asr = [99.74, 99.18, 99.16, 98.97, 98.40, 97.86, 99.26, 98.28, 98.35, 97.43, 97.48, 97.45, 97.01, 96.40, 95.65, 96.08, 95.25, 93.12, 92.59, 93.18, 95.32, 91.53, 92.54, 91.26, 88.98, 82.98, 84.33, 78.45, 77.24, 70.44, 67.89]
                else:
                    raise ValueError
            elif defense_features == 8:
                if update_mode == "top_only_[1]":
                    main_task_acc = [81.25, 81.37, 81.14, 80.06, 78.97, 76.25, 70.93, 64.97, 58.10, 51.58, 44.62, 39.26, 34.44, 30.92, 28.54, 26.50, 24.38, 22.22, 20.74, 20.06, 18.97, 18.40, 17.30, 15.65, 15.31, 15.30, 14.24, 14.42, 13.64, 12.98, 12.66]
                    asr = [92.84, 93.42, 93.59, 95.98, 97.38, 98.09, 98.55, 97.38, 96.95, 95.47, 95.29, 97.20, 98.35, 99.33, 99.35, 99.43, 99.05, 98.47, 97.89, 97.45, 97.53, 97.02, 97.06, 96.68, 96.86, 96.70, 96.83, 96.26, 96.92, 97.57, 96.84]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [81.25, 81.40, 81.20, 80.81, 79.18, 76.43, 69.46, 62.33, 56.43, 50.18, 44.65, 39.73, 35.80, 32.49, 29.87, 27.86, 26.65, 24.87, 23.68, 22.90, 22.75, 21.61, 19.93, 19.42, 19.04, 17.69, 16.62, 17.00, 15.23, 15.28, 14.67]
                    asr = [92.84, 94.98, 95.51, 97.00, 97.36, 94.42, 90.35, 76.15, 60.33, 36.30, 26.07, 28.99, 35.32, 32.30, 39.31, 37.74, 47.87, 45.70, 45.42, 40.68, 46.13, 46.33, 32.63, 34.79, 33.73, 30.97, 20.35, 31.28, 22.69, 17.00, 9.38]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [81.25, 81.50, 81.42, 81.00, 80.44, 78.91, 76.37, 71.98, 67.07, 62.15, 57.63, 54.41, 51.91, 47.38, 40.83, 38.64, 35.42, 33.06, 31.62, 30.91, 30.53, 28.50, 27.24, 26.57, 25.14, 23.93, 23.36, 22.07, 21.48, 21.58, 21.65]
                    asr = [92.84, 93.68, 94.25, 94.63, 93.63, 92.69, 94.35, 93.56, 94.76, 96.21, 95.48, 96.94, 97.84, 97.22, 96.93, 97.87, 97.31, 97.32, 96.60, 96.74, 97.00, 97.47, 95.16, 94.48, 90.73, 87.57, 85.64, 76.76, 76.54, 75.93, 74.57]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [81.25, 81.35, 81.48, 81.39, 81.53, 81.39, 81.51, 81.42, 81.44, 81.23, 81.44, 81.33, 81.05, 80.92, 80.98, 81.02, 80.75, 80.60, 80.67, 80.63, 80.35, 80.39, 80.47, 80.38, 80.32, 80.16, 80.02, 79.77, 79.73, 79.32, 79.41]
                    asr = [92.84, 93.22, 93.93, 94.33, 93.72, 92.89, 93.65, 93.78, 93.35, 92.73, 93.07, 93.09, 92.91, 92.98, 92.92, 93.25, 93.02, 92.38, 93.54, 92.85, 92.35, 91.11, 91.29, 92.07, 91.91, 90.34, 90.20, 88.04, 89.19, 86.20, 84.93]
                else:
                    raise ValueError
            elif defense_features == 12:
                if update_mode == "top_only_[1]":
                    main_task_acc = [78.15, 78.38, 78.03, 77.46, 76.56, 73.63, 68.85, 62.22, 54.72, 49.86, 45.59, 42.82, 40.30, 38.38, 35.74, 33.51, 31.30, 28.80, 27.40, 25.71, 23.81, 22.85, 21.29, 20.06, 19.18, 18.60, 17.88, 17.22, 16.23, 15.74, 15.99]
                    asr = [92.93, 96.40, 94.56, 91.71, 75.25, 55.57, 20.13, 8.71, 4.67, 2.04, 1.36, 0.62, 0.18, 0.08, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                elif update_mode == "top_only_[2]":
                    main_task_acc = [78.15, 78.48, 78.01, 76.78, 74.22, 69.04, 64.36, 60.27, 56.27, 51.09, 47.98, 44.68, 42.58, 39.42, 37.54, 35.87, 34.56, 33.01, 31.90, 31.13, 28.92, 28.88, 27.06, 26.82, 25.46, 24.33, 23.20, 22.81, 21.70, 21.95, 22.00]
                    asr = [92.93, 96.00, 98.18, 98.11, 95.78, 85.38, 70.16, 58.09, 48.65, 41.30, 37.19, 38.96, 42.70, 47.71, 57.11, 51.33, 54.19, 45.47, 35.78, 29.85, 25.51, 10.70, 9.09, 3.57, 2.85, 2.09, 0.40, 0.21, 0.19, 0.08, 0.00]
                elif update_mode == "top_only_[3]":
                    main_task_acc = [78.15, 78.54, 78.37, 78.17, 77.17, 76.56, 74.91, 72.48, 69.76, 67.05, 62.63, 60.45, 58.11, 56.30, 54.22, 52.56, 51.85, 51.10, 49.60, 47.13, 45.73, 43.57, 41.37, 37.66, 34.13, 31.63, 29.54, 27.35, 27.26, 26.22, 25.11]
                    asr = [92.93, 96.12, 96.10, 94.42, 85.41, 80.43, 71.31, 67.35, 55.85, 50.86, 41.45, 44.90, 36.47, 21.13, 5.62, 6.40, 6.02, 7.51, 9.72, 14.27, 14.76, 17.26, 20.56, 23.62, 25.46, 33.34, 49.83, 66.57, 79.23, 95.01, 99.18]
                elif update_mode == "top_only_[4]":
                    main_task_acc = [78.15, 78.70, 78.46, 78.89, 78.45, 78.54, 78.52, 78.36, 78.59, 78.47, 78.46, 78.24, 78.49, 78.10, 77.97, 78.15, 77.67, 77.78, 77.45, 77.52, 77.08, 77.35, 77.08, 76.99, 76.40, 76.29, 76.51, 76.36, 75.58, 75.18, 75.38]
                    asr = [ 92.93, 96.93, 96.09, 96.42, 96.71, 96.66, 96.84, 94.77, 96.51, 94.31, 95.78, 94.87, 94.98, 93.16, 92.95, 91.99, 89.67, 89.04, 87.24, 83.68, 78.96, 77.47, 71.19, 67.85, 53.93, 43.29, 32.30, 20.07, 9.15, 3.54, 1.17]
                else:
                    raise ValueError
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        raise ValueError
    
    return main_task_acc, asr


def plot_acc(main_task_acc, asr, 
             attack_method="TECB", dataset="cifar10", defense_features=16, update_mode="top_only_[1]",
             save_dir="../", pic_type = "png"):
    # 样式参数设置
    linewidth = 1.8
    labelsize = 18
    bwith = 2
    lwith = 2
    markevery=2
    markersize = 8
    ticksize = 12
    x_ticks_gap=5
    legendsize = 16
    legend_loc = "lower left"

    # 创建图形
    fig, ax = plt.subplots(1, 1)# figsize=(16, 9)
   
    # 设置轴的样式
    for spine in ax.spines.values():
        spine.set_linewidth(bwith)
    ax.grid(which="major", ls="--", lw=lwith, c="gray")

    # 颜色映射
    # marker 类型: o: 圆形, *: 五角星, v: 三角形, s: 正方形
    # google 配色: #f4433c 红色, #ffbc32 黄色, #0aa858 绿色, #2d85f0 蓝色
    colors = ['#0aa858', '#f4433c', '#2d85f0', '#ffbc32']
    markers = ['v', 'o', 's', '*']
    
    epochs = list(range(len(main_task_acc)))
    main_task_acc = [acc/100 for acc in main_task_acc]
    asr = [acc/100 for acc in asr]
    
    # 绘制主任务准确率
    ax.plot(
        epochs,
        main_task_acc,
        label=f'Main Task Acc',
        ls="--",
        linewidth=linewidth,
        c=colors[0],
        marker=markers[0],
        markersize=markersize,
        markevery=markevery,
    )
    # 绘制ASR
    ax.plot(
        epochs,
        asr,
        label=f'ASR',
        ls="-",
        linewidth=linewidth,
        c=colors[1],
        marker=markers[1],
        markersize=markersize,
        markevery=markevery,
    )
    
    # 设置x轴
    x_ticks = list(range(0, len(epochs) + 1, x_ticks_gap))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x}" for x in x_ticks], fontsize=ticksize)

    # 设置y轴
    ax.set_ylim(-0.05, 1.05)
    y_ticks = np.arange(0, 1.05, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.1f}" for y in y_ticks], fontsize=ticksize)

    # 设置标签和标题
    ax.set_xlabel("Epochs", fontsize=labelsize)
    ax.set_ylabel("Accuracy", fontsize=labelsize)
    
    # 添加Attack Method和Update Mode信息
    # title = f"Attack Method: TECB; Update Mode: bottom_only; Attack Feaure: {attack_features}"
    title = f"Defense Feature: {defense_feat}; Update {update_mode}"
    plt.title(title, fontsize=labelsize-2)
    
    ax.legend(fontsize=legendsize, loc=legend_loc,)
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_fig = os.path.join(save_dir, f"{attack_method}_{dataset}_defense_feat_{defense_features}_{update_mode}.{pic_type}")
    plt.savefig(save_fig, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    # attack_methods = ["TECB", "BadVFL", "Villain"]
    attack_methods = ["BadVFL", "Villain"]
    
    datasets = ["cifar10"]
    defense_features = [4, 8, 12]
    update_modes = [
        "top_only_[1]",
        "top_only_[2]",
        "top_only_[3]",
        "top_only_[4]",
    ]
    
    for attack_method in attack_methods:
        for dataset in datasets:
            for defense_feat in defense_features:
                for update_mode in update_modes:
                    main_task_acc, asr = get_data(attack_method, dataset, defense_feat, update_mode)
                    plot_acc(main_task_acc, asr, 
                            attack_method, dataset, defense_feat, update_mode,
                            save_dir="../../results/pretest/permuted_train/update_top_only", pic_type = "png")