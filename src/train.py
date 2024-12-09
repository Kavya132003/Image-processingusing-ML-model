from data_preprocessing import prepare_data
from model import create_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def train_model(data_dir, epochs=200, batch_size=8):
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_dir, test_size=0.2)
    
    # Create model
    input_shape = X_train.shape[1:]
    num_classes = len(set(y_train))
    model = create_model(input_shape, num_classes)
    
    # Enhanced data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.5),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomTranslation(0.3, 0.3),
        tf.keras.layers.RandomBrightness(0.4),
        tf.keras.layers.RandomContrast(0.4),
        tf.keras.layers.GaussianNoise(0.1)
    ])
    
    # Early stopping with longer patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        min_delta=0.01
    )
    
    # Learning rate scheduler with warmap
    initial_learning_rate = 0.00005
    warmup_epochs = 5
    
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return initial_learning_rate * ((epoch + 1) / warmup_epochs)
        else:
            return lr * (0.95 ** ((epoch - warmup_epochs) // 8))
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train with class weights to handle imbalance
    class_weights = {}
    total_samples = len(y_train)
    for class_idx in range(num_classes):
        n_samples = np.sum(y_train == class_idx)
        class_weights[class_idx] = (1 / n_samples) * (total_samples / 2)
    
    # Train model
    history = model.fit(
        data_augmentation(X_train),
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=True,
        callbacks=[early_stopping, lr_scheduler, checkpoint],
        class_weight=class_weights
    )
    
    # Load the best model
    model = tf.keras.models.load_model('best_model.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Print final metrics
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Accuracy: {final_train_accuracy:.2f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}")
    
    return model

if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Aishwarya G\OneDrive\Desktop\AICTE ML Model\data"
    train_model(DATA_DIR) 