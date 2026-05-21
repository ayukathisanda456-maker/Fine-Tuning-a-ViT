#Importing necessary libraries
import torch
import torchvision
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer,AutoModelForImageClassification, TrainerCallback
from transformers import CLIPProcessor,CLIPModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#function
def collate_fn(examples):
    pixels = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])

    return {"pixel_values": pixels, "labels": labels}

# def compute_metrics(eval_pred):
#     predictions,labels = eval_pred
#     predictions = np.argmax(predictions,axis=1)

#     return dict(accuracy = accuracy_score(predictions,labels))

def compute_metrics(eval_pred): 
    if hasattr(eval_pred, "predictions"):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        preds, labels = eval_pred

    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

class EpochAccCallback(TrainerCallback):
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_curve = []  # list of (epoch, acc)
        self.test_curve = []
        self.trainer = None  # will be set after Trainer is created

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     if self.trainer is None:
    #         return control

    #     epoch = int(round(state.epoch)) if state.epoch is not None else None

    #     train_metrics = self.trainer.evaluate(
    #         eval_dataset=self.train_dataset,
    #         metric_key_prefix="train"
    #     )
    #     test_metrics = self.trainer.predict(
    #         test_dataset=self.test_dataset,
    #         metric_key_prefix="test"
    #     ).metrics

    #     self.train_curve.append((epoch, train_metrics.get("train_accuracy")))
    #     self.test_curve.append((epoch, test_metrics.get("test_accuracy")))
    #     return control
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        epoch = int(round(state.epoch)) if state.epoch is not None else None

        old_eval_bs = self.trainer.args.per_device_eval_batch_size
        try:
            # temporarily increase eval/predict batch size
            self.trainer.args.per_device_eval_batch_size = 5000

            # force dataloader rebuilds with the new batch size if cached
            if hasattr(self.trainer, "_eval_dataloader"):
                self.trainer._eval_dataloader = None
            if hasattr(self.trainer, "_test_dataloader"):
                self.trainer._test_dataloader = None

            train_metrics = self.trainer.evaluate(
                eval_dataset=self.train_dataset,
                metric_key_prefix="train",
            )

            test_metrics = self.trainer.predict(
                test_dataset=self.test_dataset,
                metric_key_prefix="test",
            ).metrics

        finally:
            # restore original eval batch size 
            self.trainer.args.per_device_eval_batch_size = old_eval_bs

            # clear again so future eval/predict uses the restored batch size
            if hasattr(self.trainer, "_eval_dataloader"):
                self.trainer._eval_dataloader = None
            if hasattr(self.trainer, "_test_dataloader"):
                self.trainer._test_dataloader = None

        self.train_curve.append((epoch, train_metrics.get("train_accuracy")))
        self.test_curve.append((epoch, test_metrics.get("test_accuracy")))
        return control

    # def on_epoch_end(self, args, state, control, **kwargs):
    #     if self.trainer is None:
    #         return control

    #     epoch = int(round(state.epoch)) if state.epoch is not None else None
 
    #     train_metrics = self.trainer.evaluate(
    #         eval_dataset=self.train_dataset,
    #         metric_key_prefix="train"
    #     )

    #     # test accuracy with a larger eval batch size
    #     old_bs = self.trainer.args.per_device_eval_batch_size
    #     try:
    #         self.trainer.args.per_device_eval_batch_size = 5000
    #         if hasattr(self.trainer, "_test_dataloader"):
    #             self.trainer._test_dataloader = None  # force rebuild with new bs
    #         test_metrics = self.trainer.predict(
    #             test_dataset=self.test_dataset,
    #             metric_key_prefix="test"
    #         ).metrics
    #     finally:
    #         self.trainer.args.per_device_eval_batch_size = old_bs
    #         if hasattr(self.trainer, "_test_dataloader"):
    #             self.trainer._test_dataloader = None  # rebuild next time with old bs

    #     self.train_curve.append((epoch, train_metrics.get("train_accuracy")))
    #     self.test_curve.append((epoch, test_metrics.get("test_accuracy")))
    #     return control
    
def write_curve_txt(path, curve):
    with open(path, "w") as f:
        f.write("epoch\taccuracy\n")
        for ep, acc in curve:
            if acc is None:
                f.write(f"{ep}\t\n")
            else:
                f.write(f"{ep}\t{acc:.6f}\n")
                    

#We create a function that runs the ViT model for cifar-10. We input into this function the number of training epochs
#For function generalisation, we will also have a variable that now changes the amount of training data vs testing data(Number of samples).
def ViT_fine_tuned(number_of_samples,no_of_training_epochs):

    dataset = load_dataset("cifar10") #load_dataset('cifar10', split='train').shuffle(seed=42)
    testing_data = dataset["test"]
    training_data_full = dataset["train"].shuffle(seed=42)
    training_data = training_data_full.select(range(number_of_samples))

    # test_split_percentage = 0.1
    # validation_split = 0.1

    # #For the purposes of using the fact that we change the size of the number of samples we load the entire dataset
    # dataset = load_dataset('cifar10', split='train').shuffle(seed=42)

    # #We select only the number of samples that we require called sub_dataset.
    # sub_dataset = dataset.select(range(number_of_samples))

    # #First split : Split the dataset into the training data and testing data
    # main_split = sub_dataset.train_test_split(test_size=test_split_percentage,seed=42)

    # #Second split, we split the training data for validation and actual training data.
    # train_val_split = main_split['train'].train_test_split(test_size=validation_split,seed=42)

    # #Enter this new dataset into new variables. These variables are the same as what we have defined for
    # #previous model to make it easy to reuse the ViT code.

    # training_data = train_val_split['train']
    # valds = train_val_split['test']
    # testing_data = main_split['test']

    # Create a dictionary to track the images and the labels of all these images

    # integer dictionary contains the label followed by what it is, can be used to convert between logits and human labels.
    integer_dictionary = dict((k, v) for k, v in enumerate(training_data.features['label'].names))
    # string dictionary does the opposite order to be fed into the training phase.
    string_dictionary = dict((v, k) for k, v in enumerate(training_data.features['label'].names))


    # Defining our pre-trained model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                              use_fast='True')

    def batch_transform(batch):
        # img.converts the image to RGB values
        images = [img.convert('RGB') for img in batch['img']]

        # this function utilises two other internal functions to do the following
        # 1. Take the images and resizes and re-normalises them. (Also seems to return a tensor of pixel values)
        inputs = processor(images=images, return_tensors='pt')

        batch['pixel_values'] = inputs['pixel_values']

        return batch

    # The function batch_transform takes our dataset and changes into a structure accepted into our model.
    training_data.set_transform(batch_transform)
    # valds.set_transform(batch_transform)
    testing_data.set_transform(batch_transform)

    # Rather than typing new classifier, I have used one that is created for fine-tuning via some research
    # also updated the number of labels as cifar-10 has only 10 classes and also input the id2labels and labels2id
    model = AutoModelForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=10,
        ignore_mismatched_sizes=True,
        id2label=integer_dictionary,
        label2id=string_dictionary
    )

    training_arguments = TrainingArguments(
        f"run_nsamples_{number_of_samples:06d}",  # File name for the directory to store training information
        eval_strategy="no",     # evaluate manually in the callback
        save_strategy='no', 
        learning_rate=2E-5,  # How fast the model learns
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=no_of_training_epochs,
        weight_decay=0.01,  # Prevents overfitting
        load_best_model_at_end=False, 
        logging_dir='logs',
        remove_unused_columns=False,
        report_to="none",
        logging_strategy="no",
    )

    acc_cb = EpochAccCallback(training_data, testing_data)

    trainer = Trainer(
        model,
        training_arguments,
        train_dataset=training_data,
        eval_dataset=None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[acc_cb],
    )

    acc_cb.trainer = trainer

    trainer.train()

    return acc_cb.train_curve, acc_cb.test_curve

 
list_of_sample_sizes = np.linspace(10, 60000, 100, dtype=int)

for nsamples in list_of_sample_sizes:
    train_curve, test_curve = ViT_fine_tuned(number_of_samples=int(nsamples), no_of_training_epochs=20)

    write_curve_txt(f"train_accuracy_nsamples{int(nsamples):06d}.txt", train_curve)
    write_curve_txt(f"test_accuracy_nsamples{int(nsamples):06d}.txt", test_curve)






