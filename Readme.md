This git is an implementation for the paper "TUN-Det: A Novel Network for Thyroid Ultrasound Nodule Detection", MICCAI2021.


This paper presents a novel one-stage detection model, TUN-Det, for thyroid nodule detection from ultrasound scans. 
The main contributions are (i) introducing Residual U-blocks (RSU) to build the backbone of our TUN-Det, and (ii) a newly designed multi-head architecture comprised of three parallel RSU variants to replace the plain convolution layers of both the classification and regression heads.
Residual blocks enable each stage of the backbone to extract both local and global features, which plays an important role in detection of nodules with different sizes and appearances.
The multi-head design embeds the ensemble strategy into one end-to-end module to improve the accuracy and robustness by fusing multiple outputs generated by diversified sub-modules. 
Experimental results conducted on 1268 thyroid nodules from 700 patients, 
show that our newly proposed RSU backbone and the multi-head architecture for classification and regression heads greatly improve the detection accuracy against the baseline model. 
Our TUN-Det also achieves very competitive results against the state-of-the-art models on overall Average Precision ($AP$) metric and outperforms them in terms of $AP_{35}$ and $AP_{50}$, which indicates its promising performance in clinical applications.

![alt text](https://user-images.githubusercontent.com/76500053/134971961-7dee5cf4-cc50-45b9-a927-497ea1074f10.png)