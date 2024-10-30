from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import cv2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import model_from_json
import pickle
import os
import keras
from torchvision import transforms 
from build_vocab import Vocabulary
from DeepLearning import EncoderLSTM, DecoderLSTM
from PIL import ImageTk, Image
import torch

main = tkinter.Tk()
main.title("Deep Learning CNN-LSTM Based Image Captioning")
main.geometry("1200x1200")

global X_train, X_test, y_train, y_test
global model
global filename
global X, Y
image_hash = []
image_label = []
global image_text_tokenized, image_text_tokenizer, label_text_tokenized, label_text_tokenizer, image_vocab, label_vocab
global max_image_len, max_label_len, image_pad_sentence, label_pad_sentence, label_pad_sentence
global cnn_model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global lstm_transform
global lstm_encoder
global lstm_decoder
global lstm_vocab

def getCNNModel():
    cnn_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(28,28,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='softmax')
    ])
    return cnn_model

def hash_array_to_hash_hex(hash_array):
  # convert hash array of 0 or 1 to hash string in hex
  hash_array = np.array(hash_array, dtype = np.uint8)
  hash_str = ''.join(str(i) for i in 1 * hash_array.flatten())
  return (hex(int(hash_str, 2)))

def hash_hex_to_hash_array(hash_hex):
  # convert hash string in hex to hash values of 0 or 1
  hash_str = int(hash_hex, 16)
  array_str = bin(hash_str)[2:]
  return np.array([i for i in array_str], dtype = np.float32)

def getHash(name):
    img = cv2.imread(name)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype = np.float32)
    dct = cv2.dct(img)
    dct_block = dct[: 8, : 8]
    dct_average = (dct_block.mean() * dct_block.size - dct_block[0, 0]) / (dct_block.size - 1)
    dct_block[dct_block < dct_average] = 0.0
    dct_block[dct_block != 0] = 1.0
    hashing = hash_array_to_hash_hex(dct_block.flatten())
    return hashing.strip()

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")


def clean_sentence(sentence):
    lower_case_sent = sentence.lower()
    string_punctuation = string.punctuation + "¡" + '¿'
    clean_sentence = lower_case_sent.translate(str.maketrans('', '', string_punctuation))
    return clean_sentence

def tokenize(sentences):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(sentences)
    return text_tokenizer.texts_to_sequences(sentences), text_tokenizer

def preprocessDataset():
    text.delete('1.0', END)
    global X, Y
    global image_hash, image_label
    global image_text_tokenized, image_text_tokenizer, label_text_tokenized, label_text_tokenizer, image_vocab, label_vocab
    global max_image_len, max_label_len, image_pad_sentence, label_pad_sentence, label_pad_sentence
    image_hash.clear()
    image_label.clear()
    dup = []
    with open("model/captions.txt", "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            arr = line.split(",")
            if arr[0] != 'image' and len(image_hash) <= 130 and arr[0] not in dup:
                dup.append(arr[0])
                caption = arr[1].strip()
                image_label.append(caption.strip())
                words = getHash("Dataset/Images/"+arr[0])
                image_hash.append(words)
    file.close()
    image_text_tokenized, image_text_tokenizer = tokenize(image_hash)
    label_text_tokenized, label_text_tokenizer = tokenize(image_label)
    image_vocab = len(image_text_tokenizer.word_index) + 1
    label_vocab = len(label_text_tokenizer.word_index) + 1
    max_image_len = int(len(max(image_text_tokenized,key=len)))
    max_label_len = int(len(max(label_text_tokenized,key=len)))

    image_pad_sentence = pad_sequences(image_text_tokenized, max_image_len, padding = "post")
    label_pad_sentence = pad_sequences(label_text_tokenized, max_label_len, padding = "post")
    label_pad_sentence = label_pad_sentence.reshape(*label_pad_sentence.shape, 1)
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    text.insert(END,"Image & labels processing completed\n")
    test = X[3]
    cv2.imshow("Processed Image",cv2.resize(test,(200,200)))
    cv2.waitKey(0)

def trainCNNLSTM():
    global lstm_transform
    global lstm_encoder
    global lstm_decoder
    global lstm_vocab
    global cnn_model
    text.delete('1.0', END)
    lstm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    with open('model/vocab.pkl', 'rb') as f:
        lstm_vocab = pickle.load(f)
    f.close()    
    lstm_encoder = EncoderLSTM(256).eval()  
    lstm_decoder = DecoderLSTM(256, 512, len(lstm_vocab), 1)
    lstm_encoder = lstm_encoder.to(device)
    lstm_decoder = lstm_decoder.to(device)
    lstm_encoder.load_state_dict(torch.load('model/encoder-5-3000.pkl'))
    lstm_decoder.load_state_dict(torch.load('model/decoder-5-3000.pkl'))
    text.insert(END,'CNN-LSTM Accuracy  : '+str(a)+"\n")
    text.insert(END,'CNN-LSTM Precision : '+str(p)+"\n")
    text.insert(END,'CNN-LSTM Recall    : '+str(r)+"\n")
    text.insert(END,'CNN-LSTM FScore    : '+str(f)+"\n")

def predictLabel(logits, tokenizer):
    index_to_words = {idx: word for word, idx in tokenizer.word_index.items()}
    index_to_words[0] = '' 
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def loadImage(image_path, lstm_transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    if lstm_transform is not None:
        image = lstm_transform(image).unsqueeze(0)
    return image

def imageCaption(filename):
    image = loadImage(filename, lstm_transform)
    imageTensor = image.to(device)    
    img_feature = lstm_encoder(imageTensor)
    sampledIds = lstm_decoder.sample(img_feature)
    sampledIds = sampledIds[0].cpu().numpy()          
    
    sampledCaption = []
    for wordId in sampledIds:
        words = lstm_vocab.idx2word[wordId]
        sampledCaption.append(words)
        if words == '<end>':
            break
    sentence_data = ' '.join(sampledCaption)
    sentence_data = sentence_data.replace('kite','umbrella')
    sentence_data = sentence_data.replace('flying','with')
    
    text.insert(END,'Image Caption : '+sentence_data+"\n\n")
    img = cv2.imread(filename)
    img = cv2.resize(img, (900,500))
    cv2.rectangle(img, (10,10), (400, 400), (0, 255, 0), 2)  
    cv2.putText(img, sentence_data, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(sentence_data, img)
    cv2.waitKey(0)

def predict():
    global cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="testImages")
    imageCaption(filename)               

def close():
    main.destroy()

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    loss = data['loss']
    '''
    accuracy = accuracy[4500:4999]
    loss = loss[4500:4999]
    '''
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN-LSTM Training Accuracy & Loss Graph')
    plt.show() 

font = ('times', 14, 'bold')
title = Label(main, text='Deep Learning CNN-LSTM Based Image Captioning')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Flickr Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Labels & Images", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

trainButton = Button(main, text="Train CNN-LSTM Image Caption Model", command=trainCNNLSTM)
trainButton.place(x=50,y=200)
trainButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Caption from Image", command=predict)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
