# WEB APP : Deep unsupervised learning methods for the identification and characterization of TCR specificity 🫁🫀

This dashboard aims to show how perform the different deep learning methods described in the paper when given a real CDR3 sequence or a personnalized one along with their v-gene and j-gene.



## Guidelines:
1. Choose a model from which the representation will be computed.
2. Type your own CDR3 sequence or click on the 'Generate' button to generate an existing CDR3 sequence at random.
3. Choose the v-gene and j-gene of your TCR from the large list provided.
4. Click on the 'Predict' button to get the predicted representation of your TCR along with the results of its clustering.

## Results:
- The prediction embedding is computed and its reducted representation is shown along with random other sequences (using UMAP).
- The group to which the TCR is mostly to belong to is displayed along with the most represented antigen of that group.
- The clusters are determined using UMAP and K-Means clustering (with cross-validation for the choice of k).


![alt text](https://github.com/yanismiraoui/dash-models/blob/master/screenshot_app.jpg)

#### PLEASE NOTE: the website can sometimes be slow to load the text and the predictions. Please wait a few seconds for the content to load. This web application is hosted on Replit and Heroku.


:link: <a  style="display: inline;"  href=""> Website of the demo
 
:link: <a  style="display: inline;"  href="https://github.com/yanismiraoui/M4R-Project-Notebooks"> Github repository of the code used in the main analysis
