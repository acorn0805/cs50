import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    #keras.utils.to_categorical(y, num_classes=None, dtype='float32')用于将整数型标签转换为one-hot编码
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    #epochs:整数，表示训练的轮数。

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {loss:.4f}")
    print(f"测试集准确率: {accuracy:.4f}")

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images=[]
    labels=[]
    for i in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(i))#os.path.join()函数用于将多个路径组合后返回一个路径,str(i)将i转换为字符串
        for img_file in os.listdir(category_dir):#os.listdir()函数用于返回指定的文件夹包含的文件或文件夹的名字的列表
            img_path = os.path.join(category_dir, img_file)
            img = cv2.imread(img_path)#cv2.imread()函数用于读取图片
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype("float32") / 255.0
            images.append(img)
            labels.append(i)
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
     # 创建序列模型
    model = tf.keras.models.Sequential()
    
    # 第一卷积层 + 池化层
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # 第二卷积层 + 池化层
    model.add(tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # 第三卷积层 + 池化层（可选，根据数据复杂度决定）
    model.add(tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    # 将多维特征图展平为一维向量
    model.add(tf.keras.layers.Flatten())
    
    # 添加全连接层（隐藏层）
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    
    # 添加Dropout层防止过拟合
    model.add(tf.keras.layers.Dropout(0.5))
    
    # 输出层，使用softmax激活函数进行多分类
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))
    
    # 编译模型
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    main()
