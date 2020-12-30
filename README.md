# Academic-Performance-Prediction
# Abstract
Student performance prediction aims to leverage student-related information to predict their future academic outcomes, which may be  beneficial to numerous educational applications, such as personalized teaching and academic early warning.We seek to  address the problem by analyzing students’ daily studying and living behavior, which is comprehensively recorded via campus smart  cards. Different from previous studies, we propose an end-to-end  student performance prediction model, namely Tri-branch CNN. We  also use the attention mechanism and Top-k focused loss strategy to further improve the accuracy of our approach. Extensive  experiments on a large-scale real-world dataset demonstrate the  potential of our approach for student performance prediction.
# Dataset

The dataset used in this paper consists of approximately 13.7 million  smart card records of 8,199 undergraduates collected from a public  university in China, which cover the period from September 2014 to  August 2015. The records reflect various types of student behavior in  campus, such as paying for meals, entering or leaving the dormitory,  and entering the library, which totally take place at 12 different  campus locations. In the experiments, we only counted the behavior  from 6am to 12pm in a day, and took one hour as a time interval.  For each student, the student’s Grade Point Average (GPA) rank  in a major is provided. We randomly selected 200,000 pairwise  comparisons between students. Of that, 70% was used for training,  and the remaining for testing.

# Files
/data preprocessing/Top-k Focused Loss.py: calculate the NDCG value of different training pairs;  
/data preprocessing/dataset make.py: process the student swipe record into a three-dimensional tensor;  
/TB-CNN model.py: the proposed TB-CNN model structure；  
/train.py: model training.

# Requirements
Python >= 3.0  
PyTorch >= 1.0  
numpy  
torchvision 

# Hyperparameter settings
Epoch: 50
BatchSize: 16
Learning Rate: 1e-5
LR_DECAY_EPOCH = 20
LR_DECAY_RATE = 0.5

# Contact
If you have any questions, please contact superzj111@163.com.
