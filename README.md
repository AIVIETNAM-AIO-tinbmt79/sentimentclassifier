# Sentiment Classifier

This project is a personal practice exercise to learn how to fine-tune a pre-trained transformer model for sentiment classification.
The goal is not only to build a working sentiment classifier, but also to understand the full fine-tuning workflow, evaluate model performance, and identify the limitations of the model through testing.

---

## Initial Goals

At the beginning of this project, I set the following learning objectives:

### 1. Understand the fine-tuning pipeline of a pre-trained model

I wanted to learn the end-to-end process of adapting a pre-trained NLP model to a downstream task, including:

* Loading and preprocessing raw text data
* Tokenizing text using a pre-trained tokenizer
* Preparing datasets for training and validation
* Fine-tuning a transformer-based model for sequence classification
* Saving and loading the trained model for inference

This pipeline is fundamental for applying transformer models to real NLP tasks.

---

### 2. Learn how to evaluate a classification model

Another important goal was to understand how to measure whether the model performs well.

I aimed to practice using common evaluation metrics such as:

* **Accuracy**
* **F1-score**
* **Validation loss**

These metrics help evaluate not only overall correctness but also the balance between precision and recall.

---

### 3. Learn how to monitor the training process

I also wanted to track how the model changes during training by logging:

* Training loss
* Validation loss
* Evaluation metrics per epoch

This helps identify whether the model is improving, overfitting, or underfitting.

---

### 4. Test the model beyond standard validation metrics

Besides relying on validation metrics, I wanted to manually create test cases to examine how the model behaves on:

* Positive/negative sentiment
* Mixed sentiment
* Negation
* Double negation
* Sarcasm

This provides a more realistic understanding of model behavior.

---

## What I Achieved

Through this project, I successfully implemented the full fine-tuning workflow for a sentiment classification task.

### Completed pipeline

I was able to:

* Load and preprocess the sentiment dataset
* Tokenize text data using a transformer tokenizer
* Prepare custom datasets for training
* Fine-tune a pre-trained transformer model
* Evaluate the model using accuracy and F1-score
* Save the trained model and tokenizer
* Run inference on custom text inputs

This gave me practical experience with the standard workflow for transformer fine-tuning.

---

### Model evaluation

I successfully applied evaluation metrics such as:

* **Accuracy**
* **Macro F1-score**

These metrics provided a clearer picture of the model’s performance during validation.

I also learned that a good validation score does not necessarily mean the model performs well on difficult real-world cases.

---

### Training monitoring

I configured logging during training to monitor loss and evaluation metrics.

This helped me observe:

* Loss reduction over time
* Validation performance after each epoch
* Model improvement throughout training

This was useful for understanding whether the training process was effective.

---

### Hard test case evaluation

After training, I manually created challenging test cases to evaluate the model on more complex linguistic patterns.

The model performed well on:

* Standard positive/negative examples
* Mixed sentiment
* Contrastive statements

However, the model struggled with:

* **Double negation**
* **Complex negation patterns**

This revealed that the model learned many useful sentiment patterns but still had weaknesses in handling more complex semantic structures.

---

## Difficulties Encountered

During the project, I faced several challenges:

---

### 1. Understanding data preprocessing

At first, I had difficulty understanding how preprocessing decisions affect model performance.

For example:

* Removing usernames and URLs
* Choosing an appropriate `max_length`
* Deciding how much text truncation is acceptable

These choices directly impact what information the model receives.

---

### 2. Monitoring logs during training

Setting up training logs and visualization tools required additional effort.

I encountered issues such as:

* Logs not being generated correctly
* Difficulty visualizing loss curves
* Misconfiguration of logging directories

This taught me the importance of validating the training environment.

---

### 3. Interpreting evaluation metrics

Initially, I focused too much on achieving good metric values.

But through manual testing, I realized:

> High validation accuracy does not guarantee robust understanding.

The model still made mistakes on difficult cases such as:

* Double negation
* Sarcasm
* Implicit sentiment

This was an important lesson in model evaluation.

---

### 4. Model limitations on complex language patterns

One major challenge was seeing the model fail on sentences like:

* “I can't say I didn't enjoy it.”
* “Great, it broke on the first day.”

Although the model performed well overall, it often misclassified these examples.

This highlighted the limitations of:

* Training data diversity
* Model capacity
* Fine-tuning on simple sentiment labels

---

## Results

The final model achieved good performance on standard sentiment classification examples and demonstrated that the fine-tuning pipeline was implemented successfully.

### Key outcomes:

* Built a complete fine-tuning pipeline for sentiment classification
* Successfully trained and evaluated a transformer classifier
* Learned how to use evaluation metrics effectively
* Identified weaknesses through hard test cases
* Gained practical understanding of model limitations

Most importantly, this project helped me move from **theoretical understanding** to **practical experience** in training and evaluating NLP models.

Although the model is not perfect, the project successfully met the original learning goals and provided a strong foundation for more advanced NLP fine-tuning tasks in the future.
