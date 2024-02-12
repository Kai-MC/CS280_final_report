# Methane Leak Detection Using Optical Flow Estimation

## Abstract
This study addresses methane emissions from natural gas systems. By employing RAFT (Recurrent All-Pairs Field Transforms) on infrared images to magnify motion from methane leaks, we demonstrate the potential for real-time methane volume detection through optical flow estimation, particularly in binary leak/non-leak tasks. The approach shows promise over traditional techniques, though challenges in detecting faint or dispersed leaks suggest further model refinement is needed.

## Methods
Detailed descriptions of the RAFT model architecture, including the feature encoder, correlation layer, and recurrent update mechanism, are provided. The methodology section also covers the model training process, the dataset used, and the evaluation metrics.

## Experiments and Results
The paper presents the model training setup, dataset, validation results, and a qualitative assessment of the model's real-world applicability. It also discusses enhancements made to improve detection accuracy and the challenges faced in detecting leaks at various distances and under different conditions.

![Results Visualization 1](https://github.com/Kai-MC/CS280_final_report/assets/100511674/4e7ffec1-fe03-49fe-9672-0deb6b62fa1d)
![Results Visualization 2](https://github.com/Kai-MC/CS280_final_report/assets/100511674/ee123c4e-a69f-4a01-85aa-55dceda2fe3e)
![Results Visualization 3](https://github.com/Kai-MC/CS280_final_report/assets/100511674/104145d5-c3ee-49d9-949a-cbd8e433a70b)
