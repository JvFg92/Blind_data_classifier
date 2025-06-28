import tensorflow as tf

def preprocess_image(image, label, image_size=(224, 224)):
    """
    Preprocess the input image and label.
    
    Args:
        image: Input image tensor.
        label: Corresponding label tensor.
        image_size: Tuple specifying the target size for the image.
    
    Returns:
        Preprocessed image and label tensors.
    """
    # Resize the image to the target size
    image = tf.image.resize(image, image_size)
    
    # Normalize the image to [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

