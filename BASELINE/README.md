# Lab AI - HPC Tools

The purpose of this task is to practice the acceleration of the training of a Deep Learning (DL) model written in Pytorch.

## Project Overview

This project focuses on fine-tuning the BERT-Base model on a dataset for question-answering, specifically the SQuAD dataset. The objective is to optimize the training process on a single GPU and measure performance, comparing results across different configurations.

## Model Selection and Dataset Sampling

For this project, we initially chose to use the BERT-Base Uncased model from Hugging Face (BERT-Base Uncased) for fine-tuning. However, to optimize for memory and computational efficiency, we decided to use DistilBERT instead. DistilBERT is a smaller, faster, and more memory-efficient version of BERT, making it more suitable for our training and demonstration purposes due to its reduced number of parameters.

## Dataset Selection and Reduction

To speed up the training process, we fine-tuned our model on a reduced version of the SQuAD dataset. We sampled a smaller subset of the dataset to achieve faster training while still maintaining the integrity of the task.

    dataset['train'] = dataset['train'].select([i for i in range(1000)])
    dataset['validation'] = dataset['validation'].select([i for i in range(250)])

By selecting 1,000 training samples and 250 validation samples, we were able to significantly reduce the execution time, making it feasible to perform multiple training iterations and evaluations within a reasonable time frame.
We fine-tuned the model over 5 epochs, ensuring that the model had sufficient time to learn from the dataset while balancing the need for a shorter overall runtime.

## Evaluation at Each Epoch

For each epoch, we use both the training and validation datasets to monitor the model's performance. Specifically, we track metrics such as training loss, validation exact match, and F1 score after each epoch to evaluate the quality of the model and its improvements.

This approach allows us to assess how well the model generalizes to unseen data at each stage of the fine-tuning process, providing insights into the model's learning curve and helping guide any necessary adjustments to the training process.



## Result :

Example Results

Epoch 1:

    Training Time: 15 seconds
    Training Loss: 0.19
    Validation Exact Match: 16.4
    Validation F1 Score: 49.43

Epoch 2:

    Training Time: 15 seconds
    Training Loss: 0.10
    Validation Exact Match: 21.6
    Validation F1 Score: 56.69

Epoch 3:

    Training Time: 15 seconds
    Training Loss: 0.07
    Validation Exact Match: 16.4
    Validation F1 Score: 53.30

Epoch 4:

    Training Time: 15 seconds
    Training Loss: 0.06
    Validation Exact Match: 19.2
    Validation F1 Score: 51.44

Epoch 5:

    Training Time: 15 seconds
    Training Loss: 0.08
    Validation Exact Match: 19.6
    Validation F1 Score: 53.52

Total Training Time: 5 minutes 36 seconds


