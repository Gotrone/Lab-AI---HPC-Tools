# Lab AI - HPC Tools - Deliverable 2: Distributed Implementation

## Project Overview
 
The objective of this deliverable is to accelerate the training of a Deep Learning model by implementing a distributed version of the baseline code from Deliverable 1. We utilize PyTorch Lightning with Distributed Data Parallel (DDP) strategy to train the DistilBERT model on the SQuAD dataset across multiple GPUs and nodes. The goal is to measure the speedup achieved through distributed training compared to the baseline single-GPU implementation.
Modifications for Distributed Implementation

## Choice of Framework and Strategy

    Framework: PyTorch Lightning is used to simplify the training loop and handle distributed training seamlessly.
    Distributed Strategy: Distributed Data Parallel (DDP) is chosen for its efficiency and scalability across multiple GPUs and nodes.

## Data Preprocessing Functions

    train_data_preprocess(examples): Tokenizes and processes training data, aligning answer spans with token positions.
    preprocess_validation_examples(examples): Processes validation data, handling offset mappings for evaluation.

## Custom Dataset Class

    DataQA: A subclass of torch.utils.data.Dataset that prepares data samples for training and validation modes.

## Custom Data Collator

    custom_data_collator(features): Defines how to collate data batches, managing both training and validation scenarios.

## Data Module

    QADataModule: Inherits from pl.LightningDataModule, handles data loading, and sets up distributed samplers for multi-GPU training.

## Lightning Module

    QAModel: Inherits from pl.LightningModule, encapsulates the model logic, including forward pass, training step, validation step, and optimizer configuration.
    Utilizes DistilBertForQuestionAnswering from Hugging Face as the base model.
    Implements evaluation metrics using the evaluate library and handles prediction post-processing.

## Main Function

    Loads and preprocesses the SQuAD dataset.
    Initializes tokenizer, datasets, data modules, and the model.
    Sets up the PyTorch Lightning Trainer with distributed data parallel (DDP) strategy for multi-node and multi-GPU training.
    Starts the training process with trainer.fit().

## SLURM Submission Script (bash.sh)

    Specifies resource allocation for SLURM:
        2 nodes, each with 2 GPUs.
        Allocates CPUs and memory per node.
        Sets the job's time limit and output log file.

    Activates the Python environment.
    Sets NCCL parameters for efficient multi-node GPU communication.
    Launches the training script using srun.

## Error Encountered

While running the script, an error occurs:

    AttributeError: module 'pytorch_lightning.utilities.data' has no attribute 'fetching'

This error arises in the QADataModule class within the train_dataloader method:


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if self.trainer.num_devices > 1 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=pl.utilities.data.fetching.default_collate,  # Using default collate for tensors
            num_workers=self.num_workers,
        )

Current Situation

    We have encountered this error and are unsure how to resolve it.