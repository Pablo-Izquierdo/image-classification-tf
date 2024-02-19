Apple weight estimation from one view RGB images
================================================

**Author:** [Pablo Izquierdo](https://github.com/Pablo-Izquierdo) (CSIC)

**Description:** This work provides a tool to train and evaluate an image regression model. The code has been modified to predict apple weights as part of the Master's thesis related to the DigitalAlimenta project.

This work is an adaptation of the original [BrainGut-WineUp](https://alimenta365.csic.es/) project It was developed by [Miriam Cobo](https://github.com/MiriamCobo) (CSIC). You can find more information about it in the [Github Repository](https://github.com/MiriamCobo/BrainGut-WineUp).

**Table of contents**
1. [Notebooks content](#notebooks_content)
2. [More info](#more-info)
3. [Acknowledgements](#acknowledgments)

## Notebooks content

You can have more info on how to interact directly with the module by examining the ./notebooks folder:

   * [sets preparation notebook](./notebooks/1.0-Sets_preparation.ipynb): Split the data into training, validation and test sets.
    
   * [bootstrap samples notebook](./notebooks/1.0.1-Bootstrapping_samples_generator.ipynb): Generate Bootstrap samples

   * [check sets distribution notebook](./notebooks/1.1-Check_Sets_distribution.ipynb): Visualize Data Splits.

   * [model training notebook](./notebooks/2.0-Model_training.ipynb): Visualize training and validation model statistics.

   * [computing predictions notebook](./notebooks/3.0-Computing_predictions.ipynb): Test the classifier on multiple images.
   
   * [computing bootstrap predictions notebook](./notebooks/3.0.1-Computing_bootstrapping.ipynb): Test the classifier on multiple bootstrapping samples (if you provided bootstrapping folder in data-files folder), and save prediction at bootstrapping predictions folder.

   * [predictions statistics notebook](./notebooks/3.1-Prediction_statistics_regression.ipynb): Make and store the predictions of the test.txt file (if you provided one). Once you have done that you can visualize the statistics of the predictions like popular metrics (Mean Abosulte Error, Root Mean Squared Error and Coefficient of Determination) and visualize violin plots of the predictions.

   * [predictions bootstrap statistics notebook](./notebooks/3.1-Pstatistics_bootstrapping.ipynb): Make and store the predictions of the bootstrapping samples folder. Once you have done that you can visualize the statistics of the predictions like popular metrics (Mean Abosulte Error, Standard deviation and Confidence interval) and visualize violin plots of the predictions.

   * [saliency maps notebook](./notebooks/3.2-Saliency_maps.ipynb): Visualize the saliency maps of the predicted images, which show what were the most relevant pixels in order to make the prediction.

   * [saliency maps ranking notebook](./notebooks/3.2.2-Saliency_ranking.ipynb): Visualize the saliency maps of the best and worst predicted images

## More info



## Acknowledgements


