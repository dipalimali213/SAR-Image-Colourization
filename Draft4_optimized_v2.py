# Import necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc  # For garbage collection

# Enable memory growth and set memory limit
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            # Set memory limit to 80% of available GPU memory
            tf.config.set_logical_device_configuration(
                device,
                [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
            )
        print('GPU memory growth enabled and memory limit set')
    except:
        print('GPU memory configuration could not be enabled')

# Define the paths to the SAR and optical images
sar_image_folder = r"C:\Users\gaura\OneDrive\Desktop\SAR\SAR final\SAR final\QXSLAB_SAROPT\sar_image_folder"
optical_image_folder = r"C:\Users\gaura\OneDrive\Desktop\SAR\SAR final\SAR final\QXSLAB_SAROPT\optical_image_folder"

# Optimized settings for RTX 3050
IMG_SIZE = (256, 256)  # Keep 256x256 for quality
BATCH_SIZE = 4        # Increased from 1 to 4 for better training
NUM_EPOCHS = 150      # Increased epochs for better convergence
MAX_PAIRS = 4000     # Keep 4000 pairs for stability

# Load dataset paths with error handling
try:
    sar_image_paths = sorted([os.path.join(sar_image_folder, f) for f in os.listdir(sar_image_folder) if f.endswith('.png')])
    optical_image_paths = sorted([os.path.join(optical_image_folder, f) for f in os.listdir(optical_image_folder) if f.endswith('.png')])
    
    if not sar_image_paths or not optical_image_paths:
        raise ValueError("No image files found in the specified folders")
        
    if len(sar_image_paths) != len(optical_image_paths):
        raise ValueError("Number of SAR and optical images do not match")
        
except Exception as e:
    print(f"Error loading image paths: {e}")
    raise

# Print initial dataset size
total_pairs = len(sar_image_paths)
print(f"Initial dataset size: {total_pairs} pairs ({total_pairs*2} total images)")

# Randomly select subset of image pairs
if total_pairs > MAX_PAIRS:
    print(f"Reducing dataset from {total_pairs} pairs to {MAX_PAIRS} pairs...")
    print(f"Total images will be reduced from {total_pairs*2} to {MAX_PAIRS*2}")
    indices = np.random.choice(total_pairs, MAX_PAIRS, replace=False)
    indices = sorted(indices)  # Sort to maintain order
    sar_image_paths = [sar_image_paths[i] for i in indices]
    optical_image_paths = [optical_image_paths[i] for i in indices]

def load_images_in_batches(sar_paths, optical_paths, batch_size=10):  # Keep batch size at 10 for loading
    """Load images in smaller batches to prevent memory overflow"""
    all_sar_images = []
    all_optical_images = []
    
    for i in range(0, len(sar_paths), batch_size):
        batch_sar_paths = sar_paths[i:i + batch_size]
        batch_optical_paths = optical_paths[i:i + batch_size]
        
        sar_images = []
        optical_images = []
        
        print(f"Loading batch {i//batch_size + 1}/{len(sar_paths)//batch_size + 1}")
        print(f"Processing pairs {i} to {min(i + batch_size, len(sar_paths))}")
        
        for idx, (sar_path, optical_path) in enumerate(zip(batch_sar_paths, batch_optical_paths)):
            try:
                sar_img = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
                optical_img = cv2.imread(optical_path, cv2.IMREAD_COLOR)
                
                if sar_img is None or optical_img is None:
                    print(f"Skipping corrupted pair at index {i + idx}")
                    continue
                    
                sar_img = cv2.resize(sar_img, IMG_SIZE)
                optical_img = cv2.resize(optical_img, IMG_SIZE)
                
                # Normalize images
                sar_img = sar_img.astype('float32') / 255.0
                optical_img = optical_img.astype('float32') / 255.0
                
                sar_images.append(sar_img)
                optical_images.append(optical_img)
                
            except Exception as e:
                print(f"Error loading image pair at index {i + idx}: {e}")
                continue
        
        if sar_images and optical_images:  # Only extend if we have valid images
            all_sar_images.extend(sar_images)
            all_optical_images.extend(optical_images)
            print(f"Successfully loaded {len(sar_images)} pairs in this batch")
        
        # Clear memory after each batch
        gc.collect()
        tf.keras.backend.clear_session()
        
    if not all_sar_images or not all_optical_images:
        raise ValueError("No valid images were loaded")
        
    return np.array(all_sar_images), np.array(all_optical_images)

print("\nLoading image pairs in batches...")
try:
    sar_images, optical_images = load_images_in_batches(sar_image_paths, optical_image_paths)
    print(f"\nTotal image pairs loaded: {len(sar_images)}")
    print(f"Total individual images: {len(sar_images) * 2}")
except Exception as e:
    print(f"Error loading images: {e}")
    raise

# Split with larger validation set for better evaluation
X_train, X_test, y_train, y_test = train_test_split(sar_images, optical_images, test_size=0.2, random_state=42)
print(f"\nTraining pairs: {len(X_train)}")
print(f"Testing pairs: {len(X_test)}")

# Add channel dimension to SAR images
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Optimized data augmentation with more aggressive transformations
datagen = ImageDataGenerator(
    rotation_range=20,  # Increased from 15
    width_shift_range=0.2,  # Increased from 0.15
    height_shift_range=0.2,  # Increased from 0.15
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # Increased range
    zoom_range=0.2  # Increased from 0.15
)

def build_unet(input_shape=(256, 256, 1)):
    """Build U-Net model with error handling"""
    try:
        inputs = layers.Input(input_shape)

        # Encoder with increased filters
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)  # Increased from 16
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)  # Increased from 32
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)  # Increased from 64
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)  # Increased from 128
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
        c4 = layers.BatchNormalization()(c4)

        # Decoder
        u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
        c5 = layers.BatchNormalization()(c5)

        u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
        c6 = layers.BatchNormalization()(c6)

        u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)
        c7 = layers.BatchNormalization()(c7)

        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c7)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        return None

# Build and compile model with error handling
print("Building and compiling model...")
try:
    unet_model = build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    if unet_model is None:
        raise ValueError("Failed to build model")
        
    unet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Increased learning rate
        loss='mse',
        metrics=['accuracy']
    )
    print("Model built and compiled successfully")
except Exception as e:
    print(f"Error in model compilation: {e}")
    raise

# Optimized callbacks with more frequent checkpoints and progress tracking
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,  # Increased from 20
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # Increased from 10
        min_lr=0.00001,
        verbose=1
    ),
    tf.keras.callbacks.TerminateOnNaN()
]

# Create optimized tf.data pipeline with error handling
def create_dataset(X, y, batch_size):
    try:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None

try:
    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
    val_dataset = create_dataset(X_test, y_test, BATCH_SIZE)
    
    if train_dataset is None or val_dataset is None:
        raise ValueError("Failed to create datasets")
except Exception as e:
    print(f"Error creating datasets: {e}")
    raise

# Train with error handling and memory management
print("Starting training...")
try:
    history = unet_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=2,
        max_queue_size=10,  # Increased from 5
        workers=2,  # Increased from 1
        use_multiprocessing=True  # Enabled multiprocessing
    )
    
    # Save the final model
    print("Saving model...")
    unet_model.save('Sar_colourization_model.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
except Exception as e:
    print(f"Training error occurred: {e}")
    try:
        unet_model.save('Sar_colourization_model_backup.keras')
        print("Saved backup model")
    except:
        print("Could not save backup model")
finally:
    gc.collect()
    tf.keras.backend.clear_session()

# Evaluate model
print("\nEvaluating model...")
try:
    test_loss, test_accuracy = unet_model.evaluate(val_dataset)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
except Exception as e:
    print(f"Error during model evaluation: {e}")

# Visualize predictions
print("\nGenerating predictions...")
def visualize_predictions(model, X, y, num_samples=3):
    try:
        predictions = model.predict(X[:num_samples])
        for i in range(num_samples):
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(X[i].squeeze(), cmap='gray')
            plt.title('SAR Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(predictions[i])
            plt.title('Predicted Color Image')
            
            plt.subplot(1, 3, 3)
            plt.imshow(y[i])
            plt.title('Ground Truth Color Image')
            
            plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")

visualize_predictions(unet_model, X_test, y_test)
print("Done!")
