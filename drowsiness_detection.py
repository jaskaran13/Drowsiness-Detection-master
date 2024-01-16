import datetime as dt
from imutils import face_utils 
from imutils.video import VideoStream
import imutils 
import dlib
import cv2 
import pygame
from scipy.spatial import distance as dist
import os 
import time
import pandas as pd
import requests
import tkinter as tk
from tkinter import ttk
from functools import partial
from threading import Thread
from ttkthemes import ThemedTk
from tkinter import messagebox
from MyConstants import *

#Initialize Pygame and load music
pygame.mixer.init()


def assure_path_exists(path):
	dir = os.path.dirname(path)
	if not os.path.exists(dir):
		os.makedirs(dir)

def eye_aspect_ratio(eye):
	# Vertical eye landmarks
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# Horizontal eye landmarks 
	C = dist.euclidean(eye[0], eye[3])

	# The EAR Equation 
	EAR = (A + B) / (2.0 * C)
	return EAR

def mouth_aspect_ratio(mouth): 
	A = dist.euclidean(mouth[13], mouth[19])
	B = dist.euclidean(mouth[14], mouth[18])
	C = dist.euclidean(mouth[15], mouth[17])

	MAR = (A + B + C) / 3.0
	return MAR

def show_frame_text(frame,text,x1,y1,fontScale,r,g,b,thickness):
	return cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (r, g, b), thickness)

def play_sound(sound):
	pygame.mixer.music.load(sound)
	pygame.mixer.music.play()

def call_API(keyword):
	base_url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
	
	params = {
		'location': '42.30517324017192,-83.05780338465743',
		'radius': 2000,
		'keyword': keyword,
		'key': 'AIzaSyAtm0AM74ZKoW8qyY7Tk1FizmyxeO8YHaU'
	}

	response = requests.get(base_url, params=params)
	
	if response.status_code == 200:
		results = response.json()
		if 'results' in results:
			return results['results']
		else:
			print(f"Error: {results.get('error_message', 'Unknown error')}")
			return None
	else:
		print("Error: {response.status_code}")
		return None

def on_search(listbox, keyword):
	def worker():
		api_data = call_API(keyword)
		
		if api_data:
			listbox.after(0, lambda: update_listbox(listbox, api_data))
		else:
			print("Failed to retrieve data from the API.")			

	Thread(target=worker).start()
	
def update_listbox(listbox, data):
    listbox.delete(0, tk.END)
    for idx, item in enumerate(data, start=1):
        name = item['name']
        rating = str(item.get('rating', 'N/A'))
        address = item.get('vicinity', 'Address not available')
        place_id = item.get('place_id', '')

        # Constructing the HTML hyperlink
        hyperlink = f'https://www.google.com/maps/place/?q=place_id:{place_id}'

        # Creating the display string with number, name, and rating on the first line
        first_line = f"{idx}. {name} - Rating: {rating}"

        # Creating the second line with indented address
        second_line = f"    Address: {address}"

        # Creating the third line with the clickable hyperlink
        third_line = f"    Link: {hyperlink}"

        # Inserting the formatted text into the listbox
        listbox.insert(tk.END, first_line)
        listbox.insert(tk.END, second_line)
        listbox.insert(tk.END, third_line)

        # Create a label with the formatted text and bold name
        formatted_text = f"<b>{idx}. {name}</b> - Rating: {rating}\n    Address: {address}\n    Link: {hyperlink}"
        label = ttk.Label(listbox, text=formatted_text, justify="left", anchor="w", compound="left", padding=5)
        listbox.insert(tk.END, "\n")

def create_button_style():
    s = ttk.Style()
    s.configure("TButton",
                font=('Helvetica', 12),
                padding=(10, 5),
                foreground="#000000",
                background="#3498db",
                bordercolor="#3498db",
                lightcolor="#3498db",
                darkcolor="#3498db",
                relief="flat")

def show_suggestion_dialog():
	if messagebox.askyesno("Suggestion", "Do you need a suggestion?"):show_suggestion()	

def show_suggestion():
	root = ThemedTk(theme="plastik")
	root.title("Search Suggestions")
	window_width = 700
	window_height = 350
	screen_width = root.winfo_screenwidth()
	screen_height = root.winfo_screenheight()
	x = (screen_width - window_width) // 2
	y = (screen_height - window_height) // 2
	root.geometry(f"{window_width}x{window_height}+{x}+{y}")

	location_label = ttk.Label(root, text="Pause & Refresh - Nearby Rest Stops Await!", anchor="center", justify="center",font=("Helvetica", 20,"bold"))
	location_label.grid(row=0, column=0, columnspan=6, padx=10, pady=10)
	listbox = tk.Listbox(root, selectbackground="#3498db", font=("Helvetica", 11),justify="left")
	listbox.grid(row=2, column=0, columnspan=6, padx=10, pady=10, sticky="nsew")
	
	for col in range(5):
		root.columnconfigure(col, weight=2)	
	root.rowconfigure(2, weight=2)
	
	create_button_style()
	
	categories = [("Gas", "gas"),("Hotel", "hotel"), ("Cafe", "cafe"), ("Parking", "parking"), ("Food", "food")]
	for col, (text, category) in enumerate(categories, start=1):
		search_button = ttk.Button(root, text=text, command=partial(on_search, listbox, category))
		search_button.grid(row=1, column=col, padx=10, pady=10)

	root.mainloop()

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("Loading Camera.....")
vs = VideoStream(0).start()
time.sleep(2) 
assure_path_exists("dataset/")

while True: 
	# Extract a frame 
	frame = vs.read()
	# Resize the frame 
	frame = imutils.resize(frame, width = 450)
	# Convert the frame to grayscale 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Detect faces 
	rects = detector(frame, 1)
	
	# Now loop over all the face detections and apply the predictor 
	for (i, rect) in enumerate(rects): 
		shape = predictor(gray, rect)
		# Convert it to a (68, 2) size numpy array 
		shape = face_utils.shape_to_np(shape)

		# Draw a rectangle over the detected face 
		(x, y, w, h) = face_utils.rect_to_bb(rect) 
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)	
		# Put a number 
		show_frame_text(frame, "Driver", x - 5, y - 5, 0.5, 0, 255, 0, 2)

		leftEye = shape[lstart:lend]
		rightEye = shape[rstart:rend] 
		mouth = shape[mstart:mend]
		# Compute the EAR for both the eyes 
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# Take the average of both the EAR
		EAR = (leftEAR + rightEAR) / 2.0
		#live datawrite in csv
		ear_list.append(EAR)

		
		ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
		# Compute the convex hull for both the eyes and then visualize it
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		# Draw the contours 
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

		MAR = mouth_aspect_ratio(mouth)
		mar_list.append(MAR/10)
		# Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
		# Thus, count the number of frames for which the eye remains closed 
		if EAR < EAR_THRESHOLD: 
			FRAME_COUNT += 1

			cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

			if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
				FRAME_COUNT = 0
				count_sleep += 1
				# Add the frame to the dataset ar a proof of drowsy driving
				cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
				play_sound('sound files/alarm.mp3')
				show_frame_text(frame, "DROWSINESS ALERT!", 90, 300, 0.9, 0, 0, 255, 2)
				
		else: 
			FRAME_COUNT = 0
		# Check if the person is yawning
		if MAR > MAR_THRESHOLD:
			count_yawn += 1
			cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
			cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
			show_frame_text(frame, "DROWSINESS ALERT!", 90, 300, 0.9, 0, 0, 255, 2)
			play_sound('sound files/warning_yawn.mp3')
		if count_sleep>=2 | count_yawn>=2:
			count_sleep=0
			show_suggestion_dialog()
		if count_yawn>=2:
			count_yawn=0
			show_suggestion_dialog()
		
			
	#total data collection for plotting
	total_ear= ear_list[:]
	total_mar= mar_list[:]
	total_ts= ts[:]
	
	#display the frame 
	cv2.imshow("Output", frame)
	key = cv2.waitKey(1) & 0xFF 

	if key == ord('q'):
		break
		
df = pd.DataFrame({"EAR" : total_ear, "MAR":total_mar,"TIME" : total_ts}).to_csv("data_from_webcam.csv", index=True)
cv2.destroyAllWindows()
vs.stop()
