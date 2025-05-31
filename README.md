# Image Caption Generator using BLIP

This project implements an image caption generator using the BLIP (Bootstrapping Language-Image Pre-training) model. The project involves training a BLIP model on a custom dataset of images and captions to generate descriptive text for images.

## Project Structure

The project is implemented in a single Jupyter Notebook (or Google Colab notebook).

## Requirements

*   Python 3.x
*   Google Colab environment (recommended for GPU access) or a local environment with PyTorch and necessary libraries.

Install the required libraries using pip:
<pre> ```pip install transformers pandas pillow torch sentence-transformers numpy scipy matplotlib tqdm kaggle ``` </pre>

## Dataset

The project uses a custom dataset provided as CSV files and image folders.

*   `train.csv`: Contains training data with image IDs and corresponding captions.
*   `test.csv`: Contains test data with image IDs.
*   `train/`: Folder containing training images.
*   `test/`: Folder containing test images.

**Note:** The dataset needs to be downloaded. The notebook includes a command to download the dataset from a Kaggle competition (assuming you have the Kaggle API set up).

## Getting Started

1.  **Clone the repository (if applicable) or open the notebook in Google Colab.**
2.  **Install the required libraries** as mentioned in the Requirements section.
3.  **Download the dataset:** If you are using the Kaggle competition dataset, ensure your Kaggle API is configured and run the cell with the `!kaggle competitions download` command.
4.  **Extract the dataset:** The notebook includes code to extract the downloaded zip file.
5.  **Run the notebook cells sequentially.** The notebook is structured to:
    *   Load necessary libraries.
    *   Download and extract the data.
    *   Load the pre-trained BLIP model and processor.
    *   Define a custom dataset class for loading images and captions.
    *   Create data loaders for training and testing.
    *   Train the BLIP model.
    *   Fine-tune specific parts of the model (text decoder and visual backbone).
    *   Perform final training on the entire model.
    *   Generate captions for the test set.
    *   Save the generated captions to a `submission.csv` file.

## Model

The project utilizes the `Salesforce/blip-image-captioning-base` model from the `transformers` library. This is a pre-trained vision-language model that is fine-tuned on the custom dataset for the image captioning task.

## Training

The training process involves several stages:

1.  **Initial Training:** The BLIP model is trained on the training dataset for a specified number of epochs.
2.  **Fine-tuning Text Decoder:** The text decoder part of the model is fine-tuned with a lower learning rate.
3.  **Fine-tuning Visual Backbone:** The last layers of the visual backbone are fine-tuned.
4.  **Final Training:** The entire model is fine-tuned for a few more epochs.

The training progress is monitored by tracking the loss over epochs.

## Evaluation

The project generates captions for the images in the test set. These captions are saved to a `results.csv` file, which can be used for evaluation based on relevant metrics (e.g., BLEU, METEOR, CIDEr) if a ground truth is available for the test set or for submission to the Kaggle competition.


## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

[Specify your chosen license here, e.g., MIT License]

## Acknowledgements

*   Salesforce Research for developing the BLIP model.
*   Hugging Face `transformers` library.
*   PyTorch library.
*   The organizers of the obss-intern-competition-2025 for providing the dataset.
