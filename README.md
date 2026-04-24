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
## Training Loop
I set up epochs is 3

The last eval is {'eval_loss': 0.36054855585098267, 'eval_accuracy': 0.8570666666666666, 'eval_f1': 0.8570576701940783, 'eval_runtime': 51.0621, 'eval_samples_per_second': 587.52, 'eval_steps_per_second': 18.37, 'epoch': 3.0}

## Results

The final model achieved good performance on standard sentiment classification examples and demonstrated that the fine-tuning pipeline was implemented successfully.
```json
With test case 1: 
test_cases = [
    {"text": "I absolutely love this product, it works perfectly.", "label": 1},
    {"text": "This is the best purchase I've made this year.", "label": 1},
    {"text": "The movie was not bad at all.", "label": 1},
    {"text": "I didn't hate it.", "label": 1},
    {"text": "Not the best, but still enjoyable.", "label": 1},
    {"text": "The product has some flaws, but overall I'm satisfied.", "label": 1},
    {"text": "It took a while to arrive, but the quality is amazing.", "label": 1},
    {"text": "The app has bugs, but I still like using it.", "label": 1},
    {"text": "I was worried at first, but it turned out great.", "label": 1},
    {"text": "The food wasn't amazing, but I liked it.", "label": 1},

    {"text": "I hate this product.", "label": 0},
    {"text": "This is the worst thing I have ever bought.", "label": 0},
    {"text": "The movie was not good.", "label": 0},
    {"text": "I didn't like it.", "label": 0},
    {"text": "It looks nice, but it works terribly.", "label": 0},
    {"text": "The design is beautiful, but the quality is awful.", "label": 0},
    {"text": "It started well, but became disappointing.", "label": 0},
    {"text": "I wanted to love it, but it was terrible.", "label": 0},
    {"text": "The service was friendly, but completely useless.", "label": 0},
    {"text": "It isn't the worst, but I regret buying it.", "label": 0},

    {"text": "Great, it broke on the first day.", "label": 0},
    {"text": "Wonderful, another bug appeared.", "label": 0},
    {"text": "Amazing, now nothing works.", "label": 0},
    {"text": "Fantastic, it crashed again.", "label": 0},

    {"text": "I thought it would be bad, but it was actually pretty good.", "label": 1},
    {"text": "I expected something great, but it was disappointing.", "label": 0},

    {"text": "Although the shipping was late, the product quality exceeded my expectations and I am very happy with it.", "label": 1},
    {"text": "Even though the packaging looked premium, the product failed within hours and I regret buying it.", "label": 0},
]
Correct is 28/28 

With test case 2:
test_cases = [
    {"text": "I can't say I didn't enjoy it.", "label": 1},
    {"text": "It's not impossible to like this.", "label": 1},
    {"text": "I don't think it's not bad.", "label": 1},
    {"text": "I wouldn't say it was good.", "label": 0},

    # =========================
    # SARCASM
    # =========================
    {"text": "Great job, now the app won't even open.", "label": 0},
    {"text": "Fantastic, another update that breaks everything.", "label": 0},
    {"text": "Lovely, my order arrived broken again.", "label": 0},
    {"text": "Amazing service, they ignored me for a week.", "label": 0},

    # =========================
    # CONTRAST SHIFT
    # =========================
    {"text": "The design is beautiful, but everything else is terrible.", "label": 0},
    {"text": "The beginning was boring, but the ending was amazing.", "label": 1},
    {"text": "It sounds promising, but it fails miserably.", "label": 0},
    {"text": "It started badly, but turned out wonderful.", "label": 1},

    # =========================
    # MIXED SENTIMENT
    # =========================
    {"text": "I love the features, but I hate the performance.", "label": 0},
    {"text": "The performance is terrible, but I still love it.", "label": 1},
    {"text": "The food was awful, but the dessert saved the night.", "label": 1},
    {"text": "The support was helpful, but the product is unusable.", "label": 0},

    # =========================
    # IMPLIED SENTIMENT
    # =========================
    {"text": "I expected better.", "label": 0},
    {"text": "I've had worse.", "label": 1},
    {"text": "That could have gone better.", "label": 0},
    {"text": "It wasn't exactly a pleasant experience.", "label": 0},

    # =========================
    # POSITIVE WORDS BUT NEGATIVE MEANING
    # =========================
    {"text": "The app is insanely good at crashing.", "label": 0},
    {"text": "This product is perfect for wasting money.", "label": 0},
    {"text": "Excellent, it broke instantly.", "label": 0},

    # =========================
    # NEGATIVE WORDS BUT POSITIVE MEANING
    # =========================
    {"text": "This movie was wicked good.", "label": 1},
    {"text": "That performance was insanely good.", "label": 1},
    {"text": "This cake is ridiculously delicious.", "label": 1},

    # =========================
    # LONG CONTEXT REVERSAL
    # =========================
    {"text": "Although I was frustrated at first, after using it for a while I ended up loving it.", "label": 1},
    {"text": "Even though the first impression was great, after a few days it became unusable.", "label": 0},
    {"text": "At first I hated the interface, but now I can't live without it.", "label": 1},
    {"text": "I thought I would love it, but it ended up being a complete waste.", "label": 0},
]
result 23/28 
...

Through these test cases, it can be observed that the model may make incorrect predictions on long or tricky inputs because the training data is relatively simple and not diverse enough

### Key outcomes:

* Built a complete fine-tuning pipeline for sentiment classification
* Successfully trained and evaluated a transformer classifier
* Learned how to use evaluation metrics effectively
* Identified weaknesses through hard test cases
* Gained practical understanding of model limitations

Most importantly, this project helped me move from **theoretical understanding** to **practical experience** in training and evaluating NLP models.

Although the model is not perfect, the project successfully met the original learning goals and provided a strong foundation for more advanced NLP fine-tuning tasks in the future.
