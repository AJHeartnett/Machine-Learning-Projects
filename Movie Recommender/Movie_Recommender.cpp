/*
#Two most ubiquitous types, Content based and Collaborative.

#Content based systems focus on each user individually.
#It looks at the items you've already expressed interest in (via 'likes' and 'ratings') and records their keywords, attributes, and tags.
#Then your profile is gradually buit with these attributes. Then your system will strat recommending items you'd like with similar attributes to the ones you've already expressed interest in.

#Collaborative systems, it recommends you items on what other similar users have expresed interest in.
#It collaborates with many users preferences to generate you a recommendation.

# APP with C++
#Build an app that can recommend movies you'd like.
#Using Amazon's machine learning library called Deep Scalable Sparse Tensor Network Engine (DSSTNE) like Destiny.

#Our model will be a neural network because depending on how deep we make it, and how much data we feed it's just gonna outperform every other model almost every time. ;)
#Then we train it in the cloud using AWS.
#DSSTNE is what Amazon built for production-use specifically to recommend products to customers that they might like.
#It's optimizes for Sparse data and multi-GPU computation.

#Data is sparse if it contains a lot of zeroes.

#Recommendations usually operate on sparse data.
#Most ML libraries implement data-parallel training, as in it splits the training data across multiple GPUs.
#Thre's definitely a trade-off between speed and accuracy.

#DSSTNE uses model-parallel training, soinstead of splitting the data across multiple GPUs, it splits the model across multiple GPUs.
#So all layers are spread out across multiple GPUs on the same server automatically for you.

#Amazon had to do this because the weight matrices it had for recommendations, that is all the mappings of users and attributes, didn't fit in the memory of a single GPU.

#When it comes to ML libraries, DSSTNE isn't general purpose as Tensorflow (no recurrent net or LSTM capabilities yet) but it is twice as fast when it comes to dealing with sparse data.

 #Methodology includes,
 #1. Collect Dataset
 #2.Build the model
 #3.Train the model
 #4.Test the model

#we're using a sample MovieLens dataset, contains user ratings for diffrent movies and their associated tags.
#NetCDF is designed for efficient serialization of a large array of numbers and it's what DSSTNE expects.
#both of these functions (input and output) generate a NetCDF file (gl_input.nc & gl_output.nc,), an index file for neurons (features_input & features_output), and an index file for features (sample_input & sample_output)
#In DSSTNE, you build model in a JSON file instead of programmatically. 
#We can see config.json file, the structure of the neural networks. It's where we set our hyper-parameters.
#Here we are creating a 3 layer feedforward NN, which means data flows one way with one 128 node hidden layer and our activation function at each node is the classic sigmoid (which turns values into probabilities)

#we can run our train function with the batch size and number of epochs as the parameters (256, 10)
#It will create a newly trained model called "gl.nc". Which can use to predict recommendations.
#let's set number of recommendation parameters to 10. This will place a newly predictions in the "recs" file.
*/

 #include "commands.h"
 #include <iostream>

 int main() {
 	retrieveDataset("https://s3-us-west-2.amazonaws.com/amazon-dsstne-samples/data/ml20m-all");
 	generateInputLayer("ml20m-all");  //converting it to NetCDF format so that ML library can read.
 	generateOutputLayer("ml20m-all");
 	train(256, 10);
 	predict(10, "ml20m-all");
 	return 0;

 }

/*
#now that we have our code ready to be complied and run. We'll upload it to AWS.
#Make sure we're in the US East (North Virginia) region. Since,
#Amazon created a preconfigured image with dependencies like CUDA and OpenMPI already setup for us in that region. 
#click on AMIs under the images directory of the left sidebar and search for the instance called (ami-d6f2e6bc).
#Let's choose GPU to speed up our training time.
#click review and launch.
#After launching, it'll prompt you to create a new key pair, download it so you have it locally.
#This will help authorize your machine to connect to AWS.

#Now that you've successfully launched GPU instance on AWS, upload the code to it and train it.
#You can use FileZilla to upload files.

#Click the site manager icon, then paste in your host name.
#Be sure to set the protocol to SFTP, then set your login type to normal and user is called ubuntu.

#Drag and drop the project into the root folder.
#Now that our code is in our EC2 instance, we can open up terminal and SSH into it.

#You can find SSH snippet for terminal under the instances section once you click the connect button.

#paste it into terminal. CD into your directory.

#Before you run the code, make sure to add MPICC and NVCC compilers to your path and make the library.
#Export MPICC and then run make. Then export NVCC.

#Now run the script.

#You'll have recommendations in your recs folder.
#The values map to movie IDs.

#You can scale the neural net accordingly depending on the size of your data.
#Recommendation engines personalize the experience for your users to enhance their experience and increases their satisfaction.  
*/































