import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_dataset(train_ds, class_names):

    image_batch, label_batch = next(iter(train_ds))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")

def model_results(history):
    plt.figure(figsize=(20,10))
    plt.subplot(221)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()


    plt.subplot(222)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

def model_loss_acc(history, test_ds):
    test_loss, test_acc = history.evaluate(test_ds, verbose=2)

    return test_loss, test_acc

def model_results_2(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

    plt.show()

def model_results_3(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def predict_labels(model, test_ds):
    for image, label in test_ds.take(1):
        predicted_classes = model.predict(image)
        predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
        correct = np.where(predicted_classes==label)[0]
        print("Found %d correct labels" % len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.figure(figsize=(10, 10))
        plt.subplot(3,3,i+1)
        plt.imshow(image[correct].numpy().astype("uint8"), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], label[correct]))
        plt.tight_layout()
    incorrect = np.where(predicted_classes!=label)[0]
    print("Found %d incorrect labels" % len(incorrect))
    for i, incorrect in enumerate(incorrect[:9]):
        plt.figure(figsize=(10,10))
        plt.subplot(3,3,i+1)
        plt.imshow(image[incorrect].numpy().astype("uint8"), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], label[incorrect]))
        plt.tight_layout()
