# WFDC-real-time-semantic-segmentation
version and environment:

pytorch 1.5

python 3.6

torchvision 0.6

numpy 1.16

opencv 4.2
# Dataset

Save the dataset in the path "/dataset/cityscapes/..." 

Other datasets are placed in corresponding folders, such as "/dataset/camvid/..."
# Train
training model:

python train.py --dataset  --max_epochs 1200 --input_size 512,1024 --lr 4.5e-2 --batch_size 6 --classes 19 --gpus 0 or 1



# Test model
python test.py --dataset   --batch_size 1 --checkpoint "The path of the trained weights"

# Predict model for cityscapes
 python predict.py --dataset --batch_size 1 --checkpoint"The path of the trained weights"
 
# FPS 
python fps.py 512,1024 --classes 19
# FLOPs and params
python flops.py   size (3,512,1024)

# Result
Training accuracy：
![image](https://github.com/haoxj123/WFDC-real-time-semantic-segmentation/blob/main/visualization/fig0.jpg)


Training loss：
![image](https://github.com/haoxj123/WFDC-real-time-semantic-segmentation/blob/main/visualization/fig1.jpg)

![image](https://github.com/haoxj123/WFDC-real-time-semantic-segmentation/blob/main/visualization/fig2.jpg)
From left to right are: original image, ground truth, the result of WFDCNet with FCS-V module, the result of our WFDCNet with FCS module.
