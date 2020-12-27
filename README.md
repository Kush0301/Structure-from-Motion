# Sparse 3D Reconstruction


### Introduction
This project was aimed at constructing a sparse 3D point cloud from a ordered set of images through a perspective-n-point pipeline given the intrinsic camera parameters. The datasets considered for this algorithm can be found at this [link](http://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html). Specifically, the code was tested on the Gustav II Adolf Statue dataset that can be found [here](https://github.com/Kush0301/Structure-from-Motion/tree/master/sfm_dataset). The images were downsampled to reduce the computation cost. A track of 3 images was made on which the pnp pipeline was implemented.

### Method Proposed

1. Acquire the first two images from the dataset, extract features using SIFT and find a good pair of matches using FLANN based KNN matcher. The obtained matches were refined by the distance ratio test as per Lowe's paper.
2. The matched sets of keypoints were used to find out the Essential Matrix which was decomposed to recover the relative rotation and translation through SVD. The new projection matrix was calculated by using the intrinisc camera parameters. The keypoints were triangulated to find the reference 3D point cloud. 
3. The 3D point cloud was projected back to the image plane and the reprojection error was calculated by taking a norm of the difference of the reprojected points on the image plane and the keypoints detected in the same image.
4. The perspective-n-point pipeline is initiated using the keypoints and the point cloud. The distortion coefficients are assumed to be zero.
5. A reference point cloud is calculated for the n+1 image by using the n and n-1 image. Initially the value of n will be 2.
6. Acquire a new image. Let this be image number n+1. Note that the value of n will start from 2 since we have already considered 2 images. Now, the common keypoints in image n-1, image n and image n+1 (the new image) are found out. The image will now be registered using the pnp pipeline. 
7. The keypoints are triangulated and the cloud is updated. The reprojection error is then calculated. The sets of keypoints that were not common are also found out. These are new sets of points that have still not been registered in the point cloud, thus they are triangulated and the reprojection error is then calculated. The total reprojection error is then displayed.
8. For obtaining the point cloud, the 3D positions of the keypoints as well as their pixel colour is required.
9. The entire process is repeated for every new image from point 5 onwards.


### Results

The sparse point cloud obtained is first cleaned and then saved. This point cloud can be visualized using MeshLab.

<p align="center">
<img width="460" height="300" src="https://github.com/Kush0301/Structure-from-Motion/blob/master/output.png?raw=true">
</p>

