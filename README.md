#Quantum Machine Learning for Modeling the Galaxy-Dark Matter Connection

##Summary
This project is intended to compare the effectiveness of quantum machine learning to classical machine learing with cosmological data. To do so predictions made by the Halo Occupation Model (HOD) [which takes a halo’s mass to predict the number of galaxies in that halo] from Delgado et al. were used as a baseline to compare the predictions from Delgado et al.'s random forest model and the results Variational Quantum Classifier model from IBM's Qiskit predicted. The data used for both the training and testing of the models was also from Delgado et al.

##Current status and planned improvements
Currently it works (but poorly) with training and testing scores of 0.5 and 0.44 respectively when trained with the mass, environment, shear, and spin of halo parameters. Several methods from Qiskit have been deprecated since I have worked on this project and there seem to be some clear areas of improvement.

1. Small batch sizes. 
Due limitations in the IBM Qiskit lab environment small training and testing sets had to be used of around 500 data points for each parameter. This is out of tens of thousands of data points and can easily lead to the model not having an accurate prediction of what things should be based on it's training.

2. Only predicting whole integer values
When comparing the VQC results to the HOD it is clear that VQC only predicts integer values when the random forest results and the accepted (HOD) values are not only integer values. Not entirely sure what the issue here is will have to look at the actual data or about how VQC makes predictions.

3. Using specially made ansatzes and feature maps
Fitting the model to the specific problem will likely improve results rather than using the very general feature map and ansatz currently.

##Additional comments 
Overall this project was mostly spent struggling with understanding both cosmology and quantum machine learning. In addition VQC just didn't work for a while... this is largely due to IBM regularly updating and changing things. This also means that many of the examples given by IBM are outdated as they can be several years old.

However, disregarding these issues it isn't too difficult to get something just working although as mentioned it can be hard to get good results. Much of this code was simply copied either from Delgado or from the Qiskit examples and then adjusted until things worked. This means it's probably not the best method of approaching VQC but should serve as a decent starting off point.


##References
Ana Maria Delgado, Digvijay Wadekar, Boryana Hadzhiyska, Sownak Bose, Lars Hernquist, Shirley Ho, Modelling the galaxy–halo connection with machine learning, Monthly Notices of the Royal Astronomical Society, Volume 515, Issue 2, September 2022, Pages 2733–2746, https://doi.org/10.1093/mnras/stac1951
