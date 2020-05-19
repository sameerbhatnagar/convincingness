from google.colab import drive
import pandas as pd
import os
import itertools
import numpy as np
import time
import datetime
import random
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup,
)

import torch

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


import data_loaders, utils

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def train_model_and_test(
    results_dir, device, all_data=None, train_dataframes=None, test_dataframes=None,
):
    max_len = 100
    training_stats_all = []
    # Tracking variables
    # predictions_all , true_labels_all = [], []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # outer cross-validation loop
    for fold, (df_train, df_test) in enumerate(zip(train_dataframes, test_dataframes)):
        print(
            "====****==== FOLD {:} / {:} ====****====".format(
                fold + 1, len(train_dataframes)
            )
        )

        sentences = ("[CLS] " + df_train["a1"] + " [SEP] " + df_train["a2"]).values
        labels = df_train["y"].values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # Do NOT Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # The DataLoader needs to know our batch size for training, so we specify it
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch
        # size of 16 or 32.
        batch_size = 32

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size,  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=batch_size,  # Evaluate with this batch size.
        )
        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(
            model.parameters(),
            lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
        )

        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        epochs = 3

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps,
        )

        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # # Measure the total training time for the whole run.
        # total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
            print("Training...")

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = utils.format_time(time.time() - t0)

                    # Report progress.
                    print(
                        "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                            step, len(train_dataloader), elapsed
                        )
                    )

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = utils.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():

                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                    )

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = utils.format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    "epoch": epoch_i + 1,
                    "Training Loss": avg_train_loss,
                    "Valid. Loss": avg_val_loss,
                    "Valid. Accur.": avg_val_accuracy,
                    "Training Time": training_time,
                    "Validation Time": validation_time,
                }
            )

        print("")
        print("Training complete!")

        print(
            "Total training took {:} (h:mm:ss)".format(
                utils.format_time(time.time() - total_t0)
            )
        )
        training_stats_all.append(training_stats)

        ### PREDICTION ON TEST SET
        # Load the dataset into a pandas dataframe.

        # Report the number of sentences.
        print("Number of test sentences: {:,}\n".format(df_test.shape[0]))

        # Create sentence and label lists
        sentences = ("[CLS] " + df_test["a1"] + " [SEP] " + df_test["a2"]).values
        labels = df_test["y"].values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=False,  # DO NOT Add '[CLS]' and '[SEP]'
                max_length=max_len,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict["input_ids"])

            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict["attention_mask"])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        # Set the batch size.
        batch_size = 32

        # Create the DataLoader.
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(
            prediction_data, sampler=prediction_sampler, batch_size=batch_size
        )

        # Prediction on test set

        print("Predicting labels for {:,} test sentences...".format(len(input_ids)))

        # Put model in evaluation mode
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Combine the results across all batches.
        flat_predictions = np.concatenate(predictions, axis=0)
        df_test["pred_score_0"] = flat_predictions[:, 0]
        df_test["pred_score_1"] = flat_predictions[:, 1]

        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        df_test["predicted_label"] = flat_predictions

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        # Calculate the MCC
        test_set_accuracy = accuracy_score(flat_true_labels, flat_predictions)

        print("Total Accuracy: %.3f" % test_set_accuracy)
        output_dir = os.path.join(results_dir, "fold-{}".format(fold))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # model.save_pretrained(output_dir)
        df_test_fn = os.path.join(output_dir, "df_test-fold-{}.csv".format(fold))
        df_test.to_csv(df_test_fn)

        # predictions_all.append(predictions)
        # true_labels_all.append(true_labels)
        print("***********")
        print("Validation fold :")
        print(fold)
        print("    DONE.")
    return
