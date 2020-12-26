#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt


dataset=[]

#Acquiring dataset
for i in range(351, 408):
    image=cv2.imread("sfm_dataset/DSC_0"+str(i)+".JPG")
    dataset.append(image)

#Camera intrinsic parameters
K=np.array([[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], 
              [0, 2398.118540286656, 628.2649953288065], 
              [0, 0, 1]])
#Setting the down sample scale to 2, need to account for changes in the intrinsic camera parameters
down_sample=2.0
K[0,0]=K[0,0]/down_sample
K[1,1]=K[1,1]/down_sample
K[0,2]=K[0,2]/down_sample
K[1,2]=K[1,2]/down_sample


#Function returns a downsampled image
def subsample(image,down_sample):
    
    down_sample=down_sample/2
    i=0
    
    while(i<down_sample):
        image=cv2.pyrDown(image)
        i+=1
    
    return image

#Function to find the point cloud
def triangulation(proj_ref,proj_next,kp1,kp2):
    
    kp1=kp1.T
    kp2=kp2.T
    cloud = cv2.triangulatePoints(proj_ref,proj_next,kp1,kp2)
    cloud/=cloud[3]
    
    return kp1,kp2,cloud

#function for Perspective-n-point pipeline
def perspective_n_point(cloud,kp1,kp2,K):
    
    diss_coeff=np.zeros((5,1))

    _,rot,trans,inliers=cv2.solvePnPRansac(cloud,kp1,K,diss_coeff,cv2.SOLVEPNP_ITERATIVE)
    inliers=inliers[:,0]
    rot,_=cv2.Rodrigues(rot)

    return rot,trans,kp2[inliers],cloud[inliers],kp1[inliers]

#Function to calculate the error in keypoint locations by reprojecting the 3d point cloud on the image plane
def reprojection_error(cloud,kp,trans_mat,K):
    rot,_=cv2.Rodrigues(trans_mat[:3,:3])
    
    kp_reprojected,_=cv2.projectPoints(cloud,rot,trans_mat[:3,3],K,distCoeffs=None)
    kp_reprojected=np.float32(kp_reprojected[:,0,:])
    
    error=cv2.norm(kp_reprojected,kp,cv2.NORM_L2)
    
    return error/len(kp_reprojected),cloud,kp_reprojected

#Returns points that were common in the 3 images and a set of points that were not common
def three_view_points(kp2,kp2_dash,kp3):
    
    #Kp2 is the set of keypoints obtained from image(n-1) and image(n)
    #kp2_dash and kp3 are the set of keypoints obtained from image(n) and image(n+1)
    
    #Find the indices of the common keypoints in the three images by comparing kp2 and kp2_dash
    index1=[]
    index2=[]
    for i in range(kp2.shape[0]):
        if (kp2[i,:]==kp2_dash).any():
            index1.append(i)
        x=np.where(kp2_dash==kp2[i,:])
        if x[0].size!=0:
            index2.append(x[0][0])
    
    #We also need to find out the keypoints that were not common 
    kp3_uncommon=[]
    kp2_dash_uncommon=[]
    
    for k in range(kp3.shape[0]):
        if k not in index2:
            kp3_uncommon.append(list(kp3[k,:]))
            kp2_dash_uncommon.append(list(kp2_dash[k,:]))
    
    index1=np.array(index1)
    index2=np.array(index2)
    kp2_dash_common=kp2_dash[index2]
    kp3_common=kp3[index2]
            
    return index1,kp2_dash_common,kp3_common,np.array(kp2_dash_uncommon),np.array(kp3_uncommon)

#Function returns a set of good matches
def get_features(image1, image2):
    
    image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    #Using SIFT feature extraction method 
    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1=sift.detectAndCompute(image1,None)
    kp2,des2=sift.detectAndCompute(image2,None)
    des1=np.float32(des1)
    des2=np.float32(des2)
    
    #Matching the features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    match = flann.knnMatch(des1,des2,k=2)
    
    #Distance ratio test as per Lowe's paper
    filtered_matches = []
    for m,n in match:
        if m.distance<0.7*n.distance:
            filtered_matches.append(m)
            
    kp1=np.float32([kp1[m.queryIdx].pt for m in filtered_matches])
    kp2=np.float32([kp2[m.trainIdx].pt for m in filtered_matches])
    
    return kp1,kp2
        

def output(final_cloud,pixel_colour):
    
    output_points=final_cloud.reshape(-1, 3) * 200
    output_colors=pixel_colour.reshape(-1, 3)
    mesh=np.hstack([output_points,output_colors])

    mesh_mean=np.mean(mesh[:,:3],axis=0)
    diff=mesh[:,:3]-mesh_mean
    distance=np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2)
    
    index=np.where(distance<np.mean(distance)+300)
    mesh=mesh[index]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    with open('sparse.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(mesh)))
        np.savetxt(f,mesh,'%f %f %f %d %d %d')
    print("Point cloud processed, cleaned and saved successfully!")

#Initial transformation, R is identity and trans is a zero vector
initial= np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

#Intial projection matrix
proj_1=np.matmul(K,initial)
trans_12=np.empty((3,4))

#Taking the initial projection as reference
proj_ref=proj_1
final_cloud=np.zeros((1,3))
pixel_colour=np.zeros((1,3))

#Downsampling images to reduce computation cost
image1=subsample(dataset[0],down_sample)
image2=subsample(dataset[1],down_sample)

#Get first set of matched features
kp1,kp2=get_features(image1,image2)

#Calculate essential matrix
E,mask=cv2.findEssentialMat(kp1,kp2,K,method=cv2.RANSAC,prob=0.999,threshold=0.4,mask=None)
kp1=kp1[mask.ravel()==1]
kp2=kp2[mask.ravel()==1]

#Recover relative translation and rotation using the essential matrix
_,rot,trans,mask=cv2.recoverPose(E,kp1,kp2,K) 
trans=trans.ravel()

#Propagate rotation and translation
trans_12[:3,:3]=np.matmul(rot,initial[:3,:3])
trans_12[:3,3]=initial[:3, 3]+np.matmul(initial[:3, :3],trans)
proj_2=np.matmul(K,trans_12)

kp1=kp1[mask.ravel()>0]
kp2=kp2[mask.ravel()>0]

#Triangulate to obtain a reference point cloud
kp1,kp2,cloud=triangulation(proj_1,proj_2,kp1,kp2)
cloud=cv2.convertPointsFromHomogeneous(cloud.T)

#Calculate the reprojection error and initiate the pnp pipeline
error,cloud,repro_pts=reprojection_error(cloud,kp2.T,trans_12,K)
Rot,trans,kp2,cloud,kp1t=perspective_n_point(cloud[:,0,:],kp1.T,kp2.T,K)

print("Reprojection Error after 2 images:",np.round(error,4))

for i in range(len(dataset)-2):
    
    if i>0:
        #Set the reference
        kp1,kp2,cloud=triangulation(proj_1,proj_2,kp1,kp2)
        kp2=kp2.T
        cloud=cv2.convertPointsFromHomogeneous(cloud.T)
        cloud=cloud[:,0,:]
    
    #Acquiring new image and finding a set of good matches
    new_image=dataset[i+2]
    new_image=subsample(new_image,down_sample)
    kp2_dash,kp3=get_features(image2,new_image)
    
    #Finding a set of keypoints that are present in all the 3 images
    index,kp2_dash_common,kp3_common,kp2_dash_uncommon,kp3_uncommon=three_view_points(kp2,kp2_dash,kp3)
    
    #The common keypoints will be inputs to the pnp pipeline
    rot,trans,kp3_common,cloud,kp2_dash_common=perspective_n_point(cloud[index],kp3_common,kp2_dash_common,K)
    
    #Calculating the new projection matrix
    trans_mat_new=np.hstack((rot,trans))
    proj_new=np.matmul(K,trans_mat_new)

    #Using the common keypoints in image (i+1) to calculate the reprojection error
    error1,cloud,kp_projected=reprojection_error(cloud,kp3_common,trans_mat_new,K)
   
    #Updating the cloud using the uncommon points as well
    kp2_dash_uncommon,kp3_uncommon,cloud = triangulation(proj_2,proj_new,kp2_dash_uncommon,kp3_uncommon)
    cloud=cv2.convertPointsFromHomogeneous(cloud.T)
    error2,cloud,kp_reprojected=reprojection_error(cloud,kp3_uncommon.T,trans_mat_new,K)
    
    #Total reprojection error 
    print("Reprojection Error after "+str(i+3)+" images:"+str(np.round(error1+error2,4))) 
    
    #Stacking the point cloud
    final_cloud=np.vstack((final_cloud,cloud[:,0,:]))
    
    #Finding and stacking pixel colours
    kp_for_intensity = np.array(kp3_uncommon, dtype=np.int32)
    colors=np.array([new_image[intensity[1], intensity[0]] for intensity in kp_for_intensity.T])
    pixel_colour=np.vstack((pixel_colour,colors)) 
    
    #Setting values for the next iteration
    proj_1=proj_2
    proj_2=proj_new
    image2=new_image
    kp1=kp2_dash
    kp2=kp3

output(final_cloud,pixel_colour)



