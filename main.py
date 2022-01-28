from cgitb import text
from tkinter import *
from tkinter import font
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageGrab
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os, time

root = Tk()
root.config(bg='black')
root.title('Digit Classifier')
root.state('zoomed')
root.iconbitmap('icon.ico')     

fontStyle = font.Font(family="Lucida Grande", size=20)
fontStyle2 = font.Font(family='Lucida Grand', size=16)
fontStyle3 = font.Font(family='Lucida Grand', size=12)
result = Label(root, text='', bg='black', fg='white', padx=10, font=fontStyle)

global input_shape
input_shape = (28, 28, 1)

mnist = tf.keras.datasets.mnist

global x_train
global y_train
    
def modeltrain():
    
    mnist = tf.keras.datasets.mnist

    global x_train
    global y_train

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], *input_shape)
    x_test = x_test.reshape(x_test.shape[0], *input_shape)
    
    global model

    model = tf.keras.models.Sequential();
    model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=6, kernel_size=[5, 5], activation="relu", strides=(1, 1), padding="same"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation="relu", strides=(1, 1), padding="same"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    global batched_input_shape
    batched_input_shape = tf.TensorShape((None , *input_shape))
    mode = model.build(input_shape=batched_input_shape)

    call_backs = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1, write_graph=True),
        tf.keras.callbacks.ModelCheckpoint(filepath="./checkpoints/model-{epoch:02d}-{val_loss:0.2f}", monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')]

    val_x = x_train[0:5000]
    val_y = y_train[0:5000]

    x_train = x_train[5000:]
    y_train = y_train[5000:]

    digits = model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=15, validation_data=(val_x, val_y), callbacks=call_backs)
    model.evaluate(x_test , y_test)
    model.save("digits.model")

try:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = tf.keras.models.load_model('digits.model')
except:
    response = messagebox.askyesno('Warning', 'The program cannot detect a pre-configured model on your device. Please check if the "digits.model" folder is in the same directory as the program or click yes to train the model (This can take up to an hour)')
    if response == 1:
        train()
    else:
        root.destroy()

#saves the present location of the mouse pointer as the starting point of the line
def getxy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

#draws a line from the previous location of the mouse pointer to its present location
def paint(event):
    global lastx, lasty
    x1, y1 = lastx, lasty
    x2, y2 = event.x, event.y
    canvas.create_rectangle(x1, y1, x2, y2, fill='black', width=20)
    draw.line((x1, y1, x2, y2), fill='black', width=35)
    lastx, lasty = event.x, event.y

#to confirm if the user wants to train the model
def train_confirm():
    response = messagebox.askyesno('Warning', 'This can take up to an hour. Are you sure you want to continue?')
    if response == 1:
        train()
    else:
        pass
    
#calls the function to train the model
def train():
    label1 = Label(root, text='Training Model...', font=fontStyle2, bg='black', fg='gray')
    label1.pack()
    modeltrain()
    label1.destroy()

#Clears the canvas and restarts the window
def clear():
    os.startfile('main.py')
    time.sleep(0.5)
    root.destroy()

#saves the hand-drawn image as a 28x28 pixel file
def save():
    filename = 'AcceptedImage28x28.png'
    image1.save("1.png")
    num = Image.open("1.png")
    num28 = num.resize((28, 28))
    num28.save("1.png")
    for x in range(1 , 2):
        img = cv2.imread(f"{x}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        img = img.reshape(*(1 , *(28 , 28)), 1)
        prediction = model.predict(img)
        final = np.argmax(prediction)
        if int(final) == 8:
            result.config(text=f'The digit is an {final}')
        else:
            result.config(text=f'The digit is a {final}')

#creating a canvas and coding mouse pointer events
l1 = Label(root, text=' ', bg='black', fg='black')
l1.pack(anchor='center', pady=4)
result.pack(pady=25)
canvas = Canvas(root, width=400, height=400, bg='gray')
canvas.pack(anchor='center', pady=5)
canvas.bind('<Button-1>', getxy)    # for left-click of mouse
canvas.bind('<B1-Motion>', paint)   # for click-and-drag mouse

#creating an Image object to save the drawing on the canvas
image1 = Image.new('RGB', (400, 400), (255, 255, 255))
draw = ImageDraw.Draw(image1)

#Creating and placing buttons on the window
trainer = Button(root, text='Train Model', command=train_confirm, bg='black', fg='white' , font = fontStyle2)
trainer.pack(pady=5)
accept = Button(root, text='Predict', command=save, bg='black', fg='white' , font = fontStyle2)
accept.pack(pady = 5)
clearer = Button(root, text='Clear', command=clear, bg='black', fg='white' , font = fontStyle2)
clearer.pack(pady = 5)

root.mainloop()
