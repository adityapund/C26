import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU

# Load data
(train_x, _), (_, _) = fashion_mnist.load_data()
train_x = (train_x / 255.) * 2 - 1
train_x = train_x.reshape(-1, 28, 28, 1)

# Create generator
generator = Sequential([
    Dense(512, input_shape=[100]),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(256),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(128),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(784),
    Reshape([28, 28, 1])
])

# Create discriminator
discriminator = Sequential([
    Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Create GAN
GAN = Sequential([generator, discriminator])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False
GAN.compile(optimizer='adam', loss='binary_crossentropy')

# Training parameters
epochs = 50
batch_size = 100
noise_shape = 100

# Training loop
for epoch in range(epochs):
    print(f"Currently on Epoch {epoch + 1}")
    for i in range(train_x.shape[0] // batch_size):
        noise = np.random.normal(size=[batch_size, noise_shape])
        gen_image = generator.predict_on_batch(noise)
        train_dataset = train_x[i * batch_size:(i + 1) * batch_size]
        train_label = np.ones(shape=(batch_size, 1))
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(train_dataset, train_label)
        train_label = np.zeros(shape=(batch_size, 1))
        d_loss_fake = discriminator.train_on_batch(gen_image, train_label)
        noise = np.random.normal(size=[batch_size, noise_shape])
        train_label = np.ones(shape=(batch_size, 1))
        discriminator.trainable = False
        d_g_loss_batch = GAN.train_on_batch(noise, train_label)

    # plotting generated images at the start and then after every 10 epoch
    if epoch % 10 == 0:
        samples = 10
        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, 100)))
        for k in range(samples):
            plt.subplot(2, 5, k + 1)
            plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

print('Training is complete')
