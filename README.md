# Pose Comparer

## Installation

1. Download one of the following pretrained openpose models:
    - COCO:
        - http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
        - https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt
    - MPI:
        - http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
        - https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_mpi_faster_4_stages.prototxt

2. Place them in the models directory on the project root

## Running the app

The program can be executed either as an API or a Desktop app:

1. Running the app in desktop mode:

```
python desktop.py --model ../models/pose_iter_440000.caffemodel --proto ../models/openpose_pose_coco.prototxt --frame ../pictures/person1.jpg --template ../pictures/person2.jpg --dataset COCO
```

2. Running the app as an api:

```
python api.py --model ../models/pose_iter_440000.caffemodel --proto ../models/openpose_pose_coco.prototxt --dataset COCO
```

### Sample output

```
python desktop.py --model ../models/pose_iter_440000.caffemodel --proto ../models/openpose_pose_coco.prototxt --frame ../pictures/person4.jpg --template ../pictures/person2.jpg --dataset COCO
   
Comparison Result: [(0.9590177080589004, ['Neck', 'RShoulder']), (0.9550641368112371, ['Neck', 'LShoulder']), (0.9942131485173553, ['RShoulder', 'RElbow']), (0.9227181254350099, ['RElbow', 'RWrist']), (0.9178385863657164, ['LShoulder', 'LElbow']), (None, ['LElbow', 'LWrist']), (0.9036583588082726, ['Neck', 'RHip']), (0.9979523745279008, ['RHip', 'RKnee']), (0.6186889309822976, ['RKnee', 'RAnkle']), (0.9309666110041757, ['Neck', 'LHip']), (0.8557118698254541, ['LHip', 'LKnee']), (0.9083627223985159, ['LKnee', 'LAnkle']), (0.9892988574946624, ['Neck', 'Nose'])]
```

**The comparison result gives a value between -1 and 1 specifying how similar a part of the pose is between the two images.**

![pic1](https://github.com/fjunqueira/pose-comparator/blob/master/samples/pic1.png)
![pic2](https://github.com/fjunqueira/pose-comparator/blob/master/samples/pic2.png)
