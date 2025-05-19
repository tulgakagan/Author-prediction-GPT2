# Who Wrote This Review? Evaluating Authorial Voice in Real vs. GPT-2 Generated Music Reviews

## Overview

This project implements a pipeline for author prediction, fine-tuning a language model (GPT-2) for review generation based on specific authors, generating new reviews, and finally, evaluating these generated reviews using a pre-trained classifier. The workflow aims to capture and replicate the stylistic nuances of different authors in generated text.

## Project Structure

The project is organized into several Jupyter notebooks and auxiliary directories:

Group_13_LT/
├── notebooks/
│   ├── 01_Author_Prediction.ipynb # Predicts author from review text
│   ├── 02_Fine_Tuning.ipynb       # Fine-tunes GPT-2 models for each author
│   ├── 03_Review_Generator.ipynb  # Generates reviews using the fine-tuned models
│   └── 04_Review_Evaluator.ipynb  # Evaluates the generated reviews
├── data/
│   ├── content.csv
│   ├── reviews.csv
│   ├── database.sqlite
│   │
│   ├── processed/                   # (created by Notebook 1)
│   │   ├── selected_authors_data.csv
│   │   └── preprocessed_data.csv
│   │
│   └── generated/                   # (written by Notebook 3)
│       └── generated_data.csv
├── vectorizer_and_classifier/ # Stores saved models from Notebook 1
│   ├── svm_word_tfidf.pkl
│   └── word_tfidf_vect.pkl
├── requirements.txt
└── README.md

## Workflow / Execution Order

The project notebooks are designed to be run in the following order:

1.  `01_Author_Prediction.ipynb`
2.  `02_Fine_Tuning.ipynb`
3.  `03_Review_Generator.ipynb`
4.  `04_Review_Evaluator.ipynb`

Each notebook details its specific inputs, processes, and outputs.

### `01_Author_Prediction.ipynb`
* Loads review data from `content.csv`, `reviews.csv`, and `database.sqlite` located in the `data/` directory.
* Preprocesses the text data (cleaning, normalization).
* Performs exploratory data analysis (EDA) on review lengths, author contributions, etc.
* Selects the top N authors for the classification task.
* Extracts features (TF-IDF, character n-grams, sentence embeddings) from the review texts.
* Trains and evaluates several classifiers (e.g., Linear SVM, Logistic Regression, Naive Bayes) for author prediction.
* Saves the best performing TF-IDF vectorizer (`word_tfidf_vect.pkl`) and SVM classifier (`svm_word_tfidf.pkl`) to the `vectorizer_and_classifier/` directory.
* Saves the preprocessed data for the selected authors to `data/processed/preprocessed_data.csv`.

### `02_Fine_Tuning.ipynb`
* Loads the `preprocessed_data.csv` from `data/processed/`.
* Sets up the GPT-2 tokenizer and model for fine-tuning.
* Fine-tunes separate GPT-2 models for each of the selected authors using their respective review texts.
* Saves the fine-tuned models locally. It is noted that the models for this project are available at [Tughi on Hugging Face](https://www.google.com/url?sa=E&source=gmail&q=https://huggingface.co/Tughi).

### `03_Review_Generator.ipynb`
* Loads the fine-tuned GPT-2 models (per author) from Hugging Face Hub or local storage.
* Generates a specified number of synthetic reviews for each author.
* Saves the generated reviews to `data/generated/generated_reviews.csv`.

### `04_Review_Evaluator.ipynb`
* Loads the `generated_reviews.csv` from `data/generated/`.
* Loads the pre-trained TF-IDF vectorizer and SVM classifier from the `vectorizer_and_classifier/` directory.
* Transforms the generated review texts using the loaded vectorizer.
* Predicts the author for each generated review using the loaded classifier, treating these predictions as `y_pred`.
* Evaluates the predictions by calculating accuracy and macro-F1 score.
* Generates and plots a confusion matrix to visualize the classification performance for the generated reviews.

## Auxiliary Directories

* **`data/`**: Contains input datasets, preprocessed data, and generated reviews.
    * `processed/preprocessed_data.csv`: Output of `01_Author_Prediction.ipynb` and input for `02_Fine_Tuning.ipynb`.
    * `generated/generated_reviews.csv`: Output of `03_Review_Generator.ipynb` and input for `04_Review_Evaluator.ipynb`.
* **`vectorizer_and_classifier/`**: Stores the saved TF-IDF vectorizer and SVM classifier from `01_Author_Prediction.ipynb`.

## Setup / Installation

To run this project, you'll likely need Python and the libraries in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```
You may also need to download NLTK resources as prompted by the notebooks (e.g., `punkt`).

### Recommended Environment:

The notebooks, especially `02_Fine_Tuning.ipynb` and `03_Review_Generator.ipynb` which involve transformer models, are best run in an environment with GPU access for reasonable execution times. **Google Colab is the suggested and intended environment for running this project**, as it provides free access to GPU resources and comes with many libraries pre-installed.

The fine-tuned models are already present on [Hugging Face (Tughi)](https://huggingface.co/Tughi). If you'd like to perform the fine-tuning again (notebook `02_Fine_Tuning.ipynb`), ensure you select a GPU runtime in Colab or have a local GPU. The scripts are generally configured to use CPU if CUDA is unavailable, but this will be significantly slower for model training and generation.

## How to Run
Ensure the initial dataset (`database.sqlite` or `content.csv` and `reviews.csv`) is placed in the `data/` directory.
Execute the Jupyter notebooks in the specified order:
`01_Author_Prediction.ipynb`
`02_Fine_Tuning.ipynb`
`03_Review_Generator.ipynb`
`04_Review_Evaluator.ipynb`
<!-- end list -->