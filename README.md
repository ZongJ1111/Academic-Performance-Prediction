# Academic-Performance-Prediction
# Abstract
Academic performance prediction aims to leverage  student-related information to predict their future academic outcomes, which is beneficial to numerous educational applications,  such as personalized teaching and academic early warning. In  this paper, we address the problem by analyzing students’ daily  behavior trajectories, which can be comprehensively tracked  with campus smartcard records. Different from previous studies,  we propose a novel Tri-Branch CNN architecture, which is  equipped with row-wise, column-wise, and depth-wise convolution and attention operations, to capture the characteristics  of persistence, regularity, and temporal distribution of student  behavior in an end-to-end manner, respectively. Also, we cast  academic performance prediction as a top-k ranking problem,  and introduce a top-k focused loss to ensure the accuracy of  identifying academically at-risk students. Extensive experiments  were carried out on a large-scale real-world dataset, and we show  that our approach substantially outperforms recently proposed  methods for academic performance prediction.
# Dataset

The dataset used in this paper were collected from a public  university in China. It consists of two types of data, including  the records of campus behavior and academic performance.  For privacy protection, all personally identifiable information  is anonymized.  
Campus Behavior: The dataset contains approximately 13.7  million campus smartcard records of 8,199 undergraduates  from 19 majors covering an entire academic year, i.e., during  2014/09/01 to 2015/08/30. The records reflect some consumption and entry-exit behaviors of students in campus, which  totally take place at 12 different locations, i.e., the laundry  room, bathroom, teaching building, printing center, office  building, library, cafeteria, school bus, supermarket, hospital,  card center, and dormitory. We only considered the students’  behaviors occurring between 6am to 12pm of a day, and  regarded each hour as a time slot.   
Academic Performance: The academic performance of each  student is measured by Grade Point Average (GPA) over the  academic year. The absolute GPA scores were converted into  the relative performance ranking of students within a major.  As mentioned before, we ranked students in ascending order of  GPA scores. In other words, students with poorer performance  were arranged at higher ranking positions.

# Files
/data preprocessing/Top-k Focused Loss.py: calculate the ∆DCG value of different training pairs;  
/TTFB-CNN model.py: the proposed TFTB-CNN model structure；  
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
LR_DECAY_EPOCH: 20  
LR_DECAY_RATE: 0.5  

# Contact
If you have any questions, please contact superzj111@163.com.
