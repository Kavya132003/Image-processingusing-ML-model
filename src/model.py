import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    # Load pre-trained ResNet50V2 model (better feature extraction)
    base_model = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tune the last 50 layers
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    # Create the model with improved architecture
    model = models.Sequential([
        # Input preprocessing
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.GaussianNoise(0.1),
        
        # Base model
        base_model,
        
        # Feature extraction and classification
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Second dense block
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Third dense block
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with better optimizer settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 