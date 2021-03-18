#ライブラリのインポート
import os
import keras
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import glob
%matplotlib inline

#分類するクラス
classes = ["inosuke","kanawo","nezuko","tanziro","zenitsu"]
nb_classes = len(classes)
#画像の大きさ
img_width, img_height = 150,150

#トレーニング用とバリデーション用の画像格納先指定（自分のパス）
train_data_dir = "./kimetsu_image/train"
validation_data_dir = "./kimetsu_image/validation"

#トレーニングデータの画像数
nb_train_samples = 250
#バリデーション用の画像数
nb_validation_samples = 150
#バッチサイズ
batch_size = 100
#エポック数
epoch = 20

#トレーンング用、バリデーション用データを生成するジェネレータ作成
train_datagen = ImageDataGenerator(rescale=1.0/255, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0/255, zoom_range=0.2, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                   color_mode="rgb", classes=classes, class_mode="categorical",
                                                   batch_size=batch_size,shuffle=True)
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                   color_mode="rgb", classes=classes, class_mode="categorical",
                                                   batch_size=batch_size,shuffle=True)

#VGG16モデルの指定
keras.applications.vgg16.VGG16(include_top=True, weights="imagenet", 
                               input_tensor=None, input_shape=None, pooling=None, classes=1000)
                               
#VGG16のロード。FC層は不要なので　include_top=False 指定
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=input_tensor)

#VGG16のFC層の作成
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16 .output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation="softmax"))

#VGG16とFC層を結合してモデルを作成
kimetsu_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))


#VGG16の重みを固定
for layer in kimetsu_model.layers[:15]:
    layer.trainable = False
    
#多クラス分類を指定
kimetsu_model.compile(optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                     loss="categorical_crossentropy", metrics=["accuracy"])

#steps_per_epochの指定
nb_points = len(train_generator)
batch_size = 100
steps_per_epoch = ceil(nb_points / batch_size)

#作成したモデルでデータの学習
history = kimetsu_model.fit_generator(train_generator,epochs=epoch, steps_per_epoch=steps_per_epoch,
                                      validation_data=validation_generator, nb_val_samples=nb_validation_samples)

#accとval_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="o")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")

#Final.pngという名前で結果を保存
plt.savefig("Final.png")
plt.show()

#モデルの保存
import joblib
joblib.dump(kimetsu_model,"kimetsu72")

model = joblib.load("kimetsu72")

data_list = ["inosuke_1.jpg","inosuke_2.jpg","inosuke_3.jpg","kanawo_1.jpg","kanawo_2.jpg","kanawo_3.jpg",
            "nezuko_1.jpg","nezuko_2.jpg","nezuko_3.jpg","tanziro_1.jpg","tanziro_2.jpg","tanziro_3.jpg",
            "zenitsu_1.jpg","zenitsu_2.jpg","zenitsu_3.jpg"]
label_list = ["inosuke","kanawo","nezuko","tanziro","zenitsu"]


for i in range(15):
    img = mpimg.imread(data_list[i])
    plt.subplot(3, 5, i+1)
    plt.imshow(img)
    plt.show()
    
    data_img = image.load_img(data_list[i], target_size=(150, 150))#画像を指定サイズで読み込み
    data_arr = image.img_to_array(data_img)
    data_arr = preprocess_input(data_arr)#画像の正規化。前処理
    data_arr = data_arr.reshape(-1, 150, 150, 3)#画像を変換
    pred = np.argmax(model.predict(data_arr) )#モデルVGG16を使ってプロダクションする
    print(label_list[pred])

kimetsu_model = joblib.load("kimetsu72")

#テスト用のコード
#画像を読み込んで予測する
def img_predict(filename):
    #画像を読み込んで4次元テンソルへ変換
    img = image.load_img(filename, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要
    x= x/255
    #表示
    plt.imshow(img)
    plt.show()
    #指数表記を禁止にする
    np.set_printoptions(suppress=True)
    
    #画像のキャラを予測
    pred = kimetsu_model.predict(x)[0]
    #結果を表示する
    print('"inosuke":0,"kanawo":1,"nezuko":2,"tanziro":3,"zenitsu":4')
    print(pred*100)
    
#テスト用の画像データ読み込み
test = glob.glob("./kimetsu_image/test/*/*")

img_predict(test[15])
