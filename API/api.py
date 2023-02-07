from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
import numpy as np
import mediapipe as mp
import pandas as pd
import librosa, cv2
from math import ceil
import matplotlib.pylab as plt
import moviepy.editor as mpy
from inaSpeechSegmenter import Segmenter

def extract_MFCC(data,sr):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
def extract_chroma_stft(data,sr):
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    return chroma_stft
def extract_mel(data,sr):
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr,n_mels=256).T, axis=0)
    return mel

def Init():
    
    global fer_interpreter
    fer_interpreter = tf.lite.Interpreter(model_path="FER_mobilenetv2_224x224.tflite")
    fer_interpreter.allocate_tensors()

    global fer_input_details, fer_output_details
    fer_input_details = fer_interpreter.get_input_details()
    fer_output_details = fer_interpreter.get_output_details()

    global mp_face_mesh ,mp_drawing, drawing_spec, mp_drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color = (0, 255, 0))

    global routes_idx
    face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
    df = pd.DataFrame(tuple(face_oval), columns = ['x', 'y'])
    routes_idx = []
    p1 = df.iloc[0]['x']
    p2 = df.iloc[0]['y']
    for i in range(0, df.shape[0]):
        obj = df[df['x'] == p2]
        p1 = obj['x'].values[0]
        p2 = obj['y'].values[0]
        routes_idx.append((p1,p2))

    global class_names, colors_list, title_list, emotion2val
    class_names = ['Angry' ,'Happy' ,'Neutral' ,'Sad', 'Others']
    colors_list = ['r','b', 'g', 'c', 'y']
    title_list = ['FER Result','SER Result','FER and SER Result']
    emotion2val = [-1,1,0,-2,0]
    
    global seg
    seg = Segmenter()
    
    global ser_interpreter
    ser_interpreter = tf.lite.Interpreter(model_path="serlite.tflite")
    ser_interpreter.allocate_tensors()

    global ser_input_details, ser_output_details
    ser_input_details = ser_interpreter.get_input_details()
    ser_output_details = ser_interpreter.get_output_details()

    global mean, sigma
    mean = np.load('mean.npy')
    sigma = np.sqrt(np.load('var.npy'))
    
Init()

async def FER_inference(path):
    FER_prob = []
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frame = 0
    sum_prob = np.array([0,0,0,0,0],dtype=np.float32)
    with mp_face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=False,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, image = cap.read()
            n_frame += 1
            n_frame %= fps
            if n_frame == fps-1:
                FER_prob.append(sum_prob/fps)
                sum_prob = np.array([0,0,0,0,0],dtype=np.float32)
            if not ret: break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                r = []
                routes = []
                src_deg = []
                for source_idx, target_idx in routes_idx:
                    source = face_landmarks.landmark[source_idx]
                    target = face_landmarks.landmark[target_idx]
                    relative_source = (int(image.shape[1] * source.x), int(image.shape[0] * source.y))
                    relative_target = (int(image.shape[1] * target.x), int(image.shape[0] * target.y))
                    routes.append(relative_source)
                    r.append(np.sqrt(relative_source[1]**2+relative_source[0]**2))
                    deg = np.arctan(relative_source[1]/relative_source[0])
                    src_deg.append(deg)
                    routes.append(relative_target)    
                    r.append(np.sqrt(relative_target[1]**2+relative_target[0]**2))
                    deg = np.arctan(relative_target[1]/relative_target[0])
                    src_deg.append(deg)
                p = (routes[16],routes[43])
                m = (p[1][1]-p[0][1])/(p[1][0]-p[0][0])
                deg_r = np.arctan(m)               
                pad_image = np.zeros((image.shape[0]*3,image.shape[1]*3,3),dtype=np.uint8)
                pad_image[image.shape[0]:image.shape[0]*2,image.shape[1]:image.shape[1]*2] = image 
                rot_mat = cv2.getRotationMatrix2D((image.shape[1],image.shape[0]),np.rad2deg(deg_r),1.0)
                pad_image = cv2.warpAffine(pad_image,rot_mat,pad_image.shape[1::-1],flags=cv2.INTER_LINEAR)
                routes_r = []
                const1 = pad_image.shape[1]/3
                const2 = pad_image.shape[0]/3
                y1,y2,x1,x2 = float('inf'),float('-inf'),float('inf'),float('-inf')
                for point_idx in range(len(routes)):
                    theta = src_deg[point_idx]-deg_r
                    x = int(r[point_idx]*np.cos(theta)+const1)
                    y = int(r[point_idx]*np.sin(theta)+const2)
                    y1,y2,x1,x2 = min(y1,y),max(y2,y),min(x1,x),max(x2,x)
                    routes_r.append((x,y))
                face = pad_image[y1:y2,x1:x2] 
                face = cv2.resize(face,(224,224))
                face = (face-127.5)/127.5
                face = np.expand_dims(face,axis=0)
                fer_interpreter.set_tensor(fer_input_details[0]['index'], face.astype(np.float32))
                fer_interpreter.invoke()
                result = fer_interpreter.get_tensor(fer_output_details[0]['index'])[0]
                sum_prob += np.concatenate((result,[0]))
            else: 
                sum_prob += np.array([0,0,0,0,0.01])
    cap.release()
    return np.array(FER_prob,dtype=np.float32)

async def SER_inference(input_video_file):
    clip = mpy.VideoFileClip(input_video_file,verbose=0)
    output_audio_file = 'temp/full_audio.wav'
    clip.audio.write_audiofile(output_audio_file,verbose=0)
    full_data, samplerate = librosa.load(output_audio_file, sr=None)
    SER_prob = np.array([[0,0,0,0,0.01]]*int(clip.duration),dtype=np.float32)
    segmentation = seg(output_audio_file)
    for result in segmentation:
        if result[0]=='male' or result[0]=='female':
            start = int(result[1]*samplerate)
            stop = int(result[2]*samplerate)
            mfcc = extract_MFCC(full_data[start:stop],samplerate)
            mel = extract_mel(full_data[start:stop],samplerate)
            chroma = extract_chroma_stft(full_data[start:stop],samplerate)
            x = np.concatenate([mel,chroma,mfcc],axis=0)
            x = np.subtract(x,mean)
            x = np.divide(x,sigma)
            ser_interpreter.set_tensor(ser_input_details[0]['index'], x.reshape(1, -1).astype(np.float32))
            ser_interpreter.invoke()
            model_result = ser_interpreter.get_tensor(ser_output_details[0]['index'])[0]
            SER_prob[int(result[1]):ceil(result[2])] =np.concatenate((model_result,[0])) 
    return SER_prob

async def inference(input_video_file):
    fer_output = await FER_inference(input_video_file)
    ser_output = await SER_inference(input_video_file)
    weighted_output = fer_output*0.6+ser_output*0.4
    weighted_output = np.argmax(weighted_output,axis=1)
    fer_output = np.argmax(fer_output,axis=1)
    ser_output = np.argmax(ser_output,axis=1)
    for idx,data in enumerate((fer_output,ser_output,weighted_output)):
        duration = len(data)
        #Bar plot
        plt.figure(figsize=(15,10),dpi=100)
        plt.title(title_list[idx])
        plt.ylabel('Time (s)')
        plt.xlabel('Emotion')

        (n_angry, n_happy, n_neutral, n_sad, n_others) = (len(data[data==0]), len(data[data==1]), len(data[data==2]), len(data[data==3]), len(data[data==4]))
        percentage = [round(x/duration*100,2) for x in (n_angry, n_happy, n_neutral, n_sad, n_others)]
        graph_info = plt.bar(class_names,(n_angry, n_happy, n_neutral, n_sad, n_others), color = colors_list)
        for idy,p in enumerate(graph_info):
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            plt.text(x+width/2,
                     y+height*1.015,
                     str(percentage[idy])+'%',
                     ha='center',
                     weight='bold')
        plt.savefig(f'out{idx}.jpeg')
        
        #line plot
        plt.figure(figsize=(15,10),dpi=100)
        plt.title(title_list[idx])
        plt.yticks([-1,1,0,-2],['Angry' ,'Happy' ,'Neutral \nand Others' ,'Sad'])
        plt.ylabel("Emotion")
        plt.xlabel("Time(s)")
        plt.minorticks_on()
        axis = list(map(lambda x:plt.axhline(x[1], xmin=0, xmax=0, color=x[0],linewidth=5),list(zip(colors_list,emotion2val))))
        for i,idy in enumerate(data):
            axis[idy] = plt.axhline(y=emotion2val[idy], xmin=i/duration, xmax=(i+1)/duration, color=colors_list[idy],linewidth=5)
        plt.axis([0, duration, -3, 2])
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.figlegend(axis, class_names, loc='upper right', bbox_to_anchor=(0.9, 0.88))
        plt.savefig(f'out{idx+3}.jpeg')

Init()

app = FastAPI()
origins = [
    "http://139.162.10.207:3000",
    "http://139.162.10.207",
    "http://58.9.110.22:31450",
    "http://cherman.trueddns.com:31450"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    result = []
    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    await inference(file.filename)
    result = []
    for i in range(6):
        img = Image.open(rf'out{i}.jpeg')     
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        result.append(img_str)
    return result




