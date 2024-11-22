import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  
    ])

def train_dp_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=1.0,  
        noise_multiplier=0.5,  
        num_microbatches=128,  
        learning_rate=0.001
    )
    
    
    model = create_model()
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  
        metrics=['accuracy']
    )
    
    
    model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test))
    
    
    model.save('dp_model')
    print("Model trained and saved successfully!")


if __name__ == '__main__':
    train_dp_model()