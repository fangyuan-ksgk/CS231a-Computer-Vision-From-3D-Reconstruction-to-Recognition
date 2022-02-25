(p1) implements Space Carving for 3D reconstruction, where the space is voxelized and then carved according to silhouette from each camera's images.
![image](https://user-images.githubusercontent.com/66006349/155142955-72bba7f7-d26e-4eab-9713-5ab6d4928832.png)

(p2) Implements Representation Learning for the task of classifying clothe images. Comparing with training a network (consisting of an embedding network and a classifier network) directly for classifying cloth labels, we train firstly on rotation classification tasks of these clothe images and use the embedding network's parameter to construct a pretrained model. The fine-tuning upon this pretrained model is much more efficient and has faster & better convergence result, especially with fewer training data on the cloth labeling classification task.
![image](https://user-images.githubusercontent.com/66006349/155685233-8daecefa-8cd2-4f18-9e30-1de648333012.png)

![image](https://user-images.githubusercontent.com/66006349/155685277-402edb2f-c75f-44b6-b5f8-a5a5964fa01b.png)

![image](https://user-images.githubusercontent.com/66006349/155685367-5d4d5f77-ee8b-4c48-811b-4688c7b9bc71.png)


The insight is that by learning how to classify the rotation angle of these images of clothes, the embedding network 'learns' to capture the features useful also for the classification of the genre of these clothes.



