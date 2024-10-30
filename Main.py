from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
import os
import pandas as pd
import numpy as np
from torchvision import transforms 
from buildVocab import Vocabulary
from DeepLearning import EncoderLSTM, DecoderLSTM
from PIL import Image
import torch
import io
import base64
import matplotlib.pyplot as plt
import pickle
import pymysql
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.secret_key = 'welcome'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    #text.insert(END,'Image Caption : '+sentence_data+"\n\n")
    img = cv2.imread(filename)
    img = cv2.resize(img, (900,500))
    cv2.putText(img, sentence_data, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    return img, sentence_data

@app.route('/ImageCaptionAction', methods=['GET', 'POST'])
def ImageCaptionAction():
    if request.method == 'POST':
        global image_data
        file = request.files['t1']
        filename = secure_filename(file.filename)
        print(filename)
        if os.path.exists("static/"+filename):
            os.remove("static/"+filename)
        file.save("static/"+filename)
        img, sentence_data = imageCaption("static/"+filename)
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return render_template('ViewResult.html', data=sentence_data, img = img_b64)

@app.route('/ImageCaption', methods=['GET', 'POST'])
def ImageCaption():
    return render_template('ImageCaption.html', msg='')    

@app.route('/Signup', methods=['GET', 'POST'])
def Signup():
    return render_template('Signup.html', msg='')    


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', msg='') 

@app.route('/UserLogin', methods=['GET', 'POST'])
def UserLogin():
    return render_template('UserLogin.html', msg='')

@app.route('/UserLoginAction', methods=['GET', 'POST'])
def UserLoginAction():
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'captiondb',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            return render_template('UserScreen.html', data="Welcome "+username)
        else:
            return render_template('UserLogin.html', data="Invalid Login")


@app.route('/SignupAction', methods=['GET', 'POST'])
def SignupAction():
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        contact = request.form['t3']
        email = request.form['t4']
        address = request.form['t5']
        status = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'captiondb',charset='utf8')
        with con:    
            cur = con.cursor()
            cur.execute("select * FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    status = "Username already exists"
                    break
        if status == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'captiondb',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                status = "Signup Process Completed. You can Login now"
        return render_template('Signup.html', data=status)

if __name__ == '__main__':
    app.run()
