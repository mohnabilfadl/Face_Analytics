import cv2
import os,base64,io
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from config import *
from PIL import Image
from os.path import exists


app=Flask(__name__)
app.secret_key = app_key
app.config['MAX_CONTENT_LENGTH'] = file_mb_max * 1024 * 1024

# Check that the upload folder exists
def makedir (dest):
    fullpath = '%s/%s'%(upload_dest,dest)
    if not os.path.isdir(fullpath):
        os.mkdir(fullpath)


## on page load display the upload file
@app.route('/')
def upload_form():
    flash('Drag files to upload here.')
    return render_template('home.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

## on a POST request of data 
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if not os.path.isdir(upload_dest):
            os.mkdir(upload_dest)

    if 'files[]' not in request.files:
            flash('No files found, try again.')
            return redirect(request.url)
    files = request.files.getlist('files[]')
    for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join( upload_dest, filename))
    flash('File(s) uploaded')

    checkGender('uploads_folder/' + filename)

    img = Image.open(os.path.join(os.getcwd(), 'uploads_folder') + "/" + "check.jpg")
    data = io.BytesIO()
    img.save(data, "JPEG")

    encode_img = base64.b64encode(data.getvalue())

    return render_template("home.html", filename=encode_img.decode("UTF-8"))


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def checkGender(path):
    faceProto="models/opencv_face_detector.pbtxt"
    faceModel="models/opencv_face_detector_uint8.pb"
    ageProto="models/age_deploy.prototxt"
    ageModel="models/age_net.caffemodel"
    genderProto="models/gender_deploy.prototxt"
    genderModel="models/gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)

    video=cv2.VideoCapture(path)
    padding=20
    while True:
        hasFrame,frame=video.read()

        resultImg,faceBoxes=highlightFace(faceNet,frame)
        if not faceBoxes:
            print("No face detected")

        faceBox = faceBoxes[0]
        face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

        img = Image.fromarray(resultImg)
        img.save(os.path.join( upload_dest, "check.jpg"))

        if exists('uploads_folder' + '/' + 'check.jpg'): break


if __name__ == "__main__":
    app.run()