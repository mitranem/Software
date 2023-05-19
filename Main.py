#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install mediapipe opencv-python msvc-runtime


# In[1]:


import mediapipe as mp
import cv2

import csv
import os
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 

from collections import Counter
import matplotlib.pyplot as plt

import customtkinter as ctk
import tkinter.messagebox as tkmb
import cv2
from PIL import Image, ImageTk

# In[2]:


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# # Feed from Camera and detection

# In[3]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# # Capture Landmark and export to CSV

# In[3]:




# In[40]:


num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
num_coords


# In[41]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# In[42]:


landmarks


# In[43]:


with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[40]:


class_name = "Angry"


# In[41]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# # Splitting into train and test

# In[42]:



# In[43]:


df = pd.read_csv('coords.csv')


# In[44]:


df.head()


# In[45]:


df.head()


# In[46]:


df.tail()


# In[47]:


df[df['class']=='Sad']


# In[48]:


X = df.drop('class', axis=1) # features
y = df['class'] # target value


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


# In[50]:


y_train


# # Training and Evaluating Model

# In[51]:




# In[52]:


pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}


# In[53]:


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model


# In[54]:


fit_models


# In[55]:


fit_models['rc'].predict(X_test)


# In[56]:




# In[57]:


for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


# In[58]:


fit_models['rf'].predict(X_test)


# In[59]:


y_test


# In[60]:


with open('body_language.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)


# # Make Detections

# In[61]:


with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


# In[62]:


model


# In[28]:


type(body_language_class.split(' ')[0])


# In[79]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
emotions=[]
writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
#             # Append class name 
#             row.insert(0, class_name)
            
#             # Export to CSV
#             with open('coords.csv', mode='a', newline='') as f:
#                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 csv_writer.writerow(row) 

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            emotions += [body_language_class.split(' ')[0]]
            
            writer.write(frame)
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()


# In[78]:

frequency = Counter(emotions)
xa=[]
ya=[]
print(frequency)
for string, frequency in frequency.items():
    xa += [string]
    ya.append(frequency)

plt.plot(xa, ya, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
  


# In[1]:

  
# Selecting GUI theme - dark, light , system (for system default)
ctk.set_appearance_mode("dark")
  
# Selecting color theme - blue, green, dark-blue
ctk.set_default_color_theme("blue")
  
app = ctk.CTk()
app.geometry("400x400")
app.title("Modern Login UI using Customtkinter")
  
def process_frames():
    _, frame = cap.read()
    # Perform any necessary image processing or analysis on the frame
    # ...

    # Display the processed frame in the GUI
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
    video_label.config(image=photo)
    video_label.image = photo

    # Call this function again after a delay to process the next frame
    window.after(10, process_frames)

# Function to start video recording
def start_recording():
    global is_recording, out
    if not is_recording:
        is_recording = True
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))

# Function to stop video recording
def stop_recording():
    global is_recording, out
    if is_recording:
        is_recording = False
        out.release()

def video():
    video_app=ctk.CTk()
    video_app.geometry("400x400")
    video_app.title("Recording page")
    vlabel = ctk.CTkLabel(app,text="This is the main recording page")
    label.pack(pady=20)
    vframe = ctk.CTkFrame(master=video_app)
    vframe.pack(pady=20,padx=40,fill='both',expand=True)
    vlabel = ctk.CTkLabel(master=vframe,text="Main recording page")
    vlabel.pack(pady=12,padx=10)
    record_button = ctk.CTkButton(master=vframe, text="Start Recording",command=start_recording)
    record_button.pack(pady=12,padx=10)
    stop_button = ctk.CTkButton(master=vframe, text="Stop Recording",command=stop_recording)
    stop_button.pack(pady=12,padx=10)
    cap = cv2.VideoCapture(0)
    is_recording = False
    out = None
    # Start processing video frames
    _, frame = cap.read()
    # Perform any necessary image processing or analysis on the frame
    # ...

    # Display the processed frame in the GUI
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(image))
    vlabel.configure(image=photo)
    vlabel.image = photo

    # Call this function again after a delay to process the next frame
    video_app.after(10, process_frames)
    cap.release
    video_app.mainloop()

def login():
  
    username = "Geeks"
    password = "12345"
    new_window = ctk.CTkToplevel(app)
  
    new_window.title("New Window")
  
    new_window.geometry("350x150")
  
    if user_entry.get() == username and user_pass.get() == password:
        tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
        video()
    elif user_entry.get() == username and user_pass.get() != password:
        tkmb.showwarning(title='Wrong password',message='Please check your password')
    elif user_entry.get() != username and user_pass.get() == password:
        tkmb.showwarning(title='Wrong username',message='Please check your username')
    else:
        tkmb.showerror(title="Login Failed",message="Invalid Username and password")
  
  
  
label = ctk.CTkLabel(app,text="This is the main UI page")
  
label.pack(pady=20)
  
  
frame = ctk.CTkFrame(master=app)
frame.pack(pady=20,padx=40,fill='both',expand=True)
  
label = ctk.CTkLabel(master=frame,text='Modern Login System UI')
label.pack(pady=12,padx=10)
  
  
user_entry= ctk.CTkEntry(master=frame,placeholder_text="Username")
user_entry.pack(pady=12,padx=10)
  
user_pass= ctk.CTkEntry(master=frame,placeholder_text="Password",show="*")
user_pass.pack(pady=12,padx=10)
  
  
button = ctk.CTkButton(master=frame,text='Login',command=login)
button.pack(pady=12,padx=10)
  
  
  
app.mainloop()


# In[ ]:




