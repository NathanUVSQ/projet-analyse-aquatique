import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import os
import time

# ----------------------------
# Variables globales
# ----------------------------
global cap
global numero_camera
global record_yes
record_yes = False
global type_record
type_record = "video"    # "video" ou "photo"

recordings = {}
recording_done = set()
photo_done = set()      # 1 seule photo par ID

start = False
previous_gray = None

tracked_objects = {}
next_id = 0

MAX_DIST = 50

FPS_ESTIMATE = 15

IMMOBILE_TIMEOUT_SEC = 5
IMMOBILE_TIMEOUT_FRAMES = FPS_ESTIMATE * IMMOBILE_TIMEOUT_SEC

# ----------------------------
# AJOUT MINIMAL VIDÉOS
# ----------------------------

RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
os.makedirs(RECORD_DIR, exist_ok=True)
print(f"[INFO] Les vidéos seront enregistrées dans : {RECORD_DIR}")

RECORD_FPS = FPS_ESTIMATE
RECORD_SIZE = (200, 200)
MAX_RECORD_FRAMES = FPS_ESTIMATE * 3
VIDEO_ZOOM_FACTOR = 2

# ----------------------------
# Initialisation caméra
# ----------------------------

cap = cv2.VideoCapture(0)

# ----------------------------
# Fenêtre Tkinter
# ----------------------------

root = tk.Tk()
root.title("Analyse Microscope")
root.geometry("1000x700")
numero_camera = tk.StringVar(value="0")
video_label = tk.Label(root)
video_label.pack()

compteur_label = tk.Label(
    root,
    text="Objets mobiles détectés : 0",
    font=("Arial", 16)
)
compteur_label.pack(pady=10)

terminal_label = tk.Label(
    root,
    text="...",
    font=("Lucida Console", 10)
)
terminal_label.pack(side="left")

# ----------------------------
# Fonction vidéo
# ----------------------------

def update_frame():

    global previous_gray
    global next_id
    global tracked_objects

    ret, frame = cap.read()

    if ret and start:

        display = frame.copy()
        fh, fw = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if previous_gray is None:
            previous_gray = gray.copy()

        else:

            diff = cv2.absdiff(previous_gray, gray)
            previous_gray = gray.copy()

            _, motion = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)
            motion = cv2.dilate(motion, kernel, iterations=5)

            contours, _ = cv2.findContours(
                motion,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            seen_this_frame = set()

            for contour in contours:

                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter > 0:
                    circularity = 4 * 3.14159 * area / (perimeter ** 2)
                else:
                    circularity = 0

                if area < 500:
                    continue

                if area < 200:
                    type_objet = "Petit"
                elif circularity > 0.8:
                    type_objet = "Rond"
                elif area > 15000:
                    type_objet = "Grand"
                else:
                    type_objet = "Non definit ???"

                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2

                best_id = None
                best_dist = 999999

                for tid, obj in tracked_objects.items():
                    dist = math.sqrt((cx - obj["x"]) ** 2 + (cy - obj["y"]) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid

                if best_id is not None and best_dist < MAX_DIST:
                    obj_id = best_id
                    obj = tracked_objects[obj_id]
                    obj["x"] = cx
                    obj["y"] = cy
                    obj["w"] = w
                    obj["h"] = h
                    obj["area"] = area
                    obj["type"] = type_objet
                    obj["lost"] = 0
                else:
                    obj_id = next_id
                    tracked_objects[obj_id] = {
                        "x": cx, "y": cy,
                        "w": w, "h": h,
                        "area": area, "type": type_objet,
                        "lost": 0
                    }
                    next_id += 1

                seen_this_frame.add(obj_id)

            for obj_id in list(tracked_objects.keys()):
                if obj_id not in seen_this_frame:
                    tracked_objects[obj_id]["lost"] += 1
                if tracked_objects[obj_id]["lost"] > IMMOBILE_TIMEOUT_FRAMES:
                    del tracked_objects[obj_id]
                    recordings.pop(obj_id, None)

            for obj_id, obj in tracked_objects.items():

                x = obj["x"] - obj["w"] // 2
                y = obj["y"] - obj["h"] // 2
                w = obj["w"]
                h = obj["h"]

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"ID:{obj_id} A:{int(obj['area'])} {obj['type']}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1
                )

                # 🎥 MODE VIDÉO
                if type_record == "video":

                    if obj_id not in recording_done:

                        new_w = int(obj["w"] * VIDEO_ZOOM_FACTOR)
                        new_h = int(obj["h"] * VIDEO_ZOOM_FACTOR)

                        vx0 = max(0, obj["x"] - new_w // 2)
                        vy0 = max(0, obj["y"] - new_h // 2)
                        vx1 = min(fw, obj["x"] + new_w // 2)
                        vy1 = min(fh, obj["y"] + new_h // 2)

                        if vx1 > vx0 and vy1 > vy0:

                            crop = frame[vy0:vy1, vx0:vx1]
                            crop_resized = cv2.resize(crop, RECORD_SIZE)

                            recordings.setdefault(obj_id, []).append(crop_resized)

                            if len(recordings[obj_id]) >= MAX_RECORD_FRAMES and record_yes:

                                filename = os.path.join(
                                    RECORD_DIR,
                                    f"obj_{obj_id}_{int(time.time()*1000)}.mp4"
                                )

                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(filename, fourcc, RECORD_FPS, RECORD_SIZE)

                                for f in recordings[obj_id]:
                                    writer.write(f)

                                writer.release()

                                print(f"[OK] Vidéo enregistrée : {filename}")

                                recording_done.add(obj_id)
                                del recordings[obj_id]

                # 📸 MODE PHOTO
                elif type_record == "photo":

                    if record_yes and obj_id not in photo_done:

                        fh2, fw2 = frame.shape[:2]
                        x0 = max(0, x)
                        y0 = max(0, y)
                        x1 = min(fw2, x + w)
                        y1 = min(fh2, y + h)

                        if x1 > x0 and y1 > y0:

                            crop = frame[y0:y1, x0:x1]

                            filename = os.path.join(
                                RECORD_DIR,
                                f"obj_{obj_id}_{int(time.time()*1000)}.png"
                            )

                            ok = cv2.imwrite(filename, crop)

                            if ok:
                                print(f"[OK] Photo enregistrée : {filename}")
                                photo_done.add(obj_id)
                            else:
                                print(f"[ERREUR] Échec photo pour ID {obj_id}")

            compteur_label.config(
                text=f"Objets mobiles détectés : {len(tracked_objects)}"
            )

            cv2.imshow("Detection du mouvement", motion)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    root.after(10, update_frame)

# ----------------------------
# Bouton démarrer / arrêter
# ----------------------------

def commencer():

    global start
    global previous_gray

    if start:
        start = False
        previous_gray = None
        cv2.destroyAllWindows()
        terminal_label.config(text="Live is stopped")
    else:
        start = True
        terminal_label.config(text="Live is launched")

btn_start = tk.Button(
    root,
    text="Live",
    command=commencer,
    font=("Arial", 14)
)
btn_start.pack(pady=10)

# ----------------------------
# record boutton
# ----------------------------

def record_YN():

    global record_yes

    if record_yes:
        record_yes = False
        terminal_label.config(text="Stop recording !")
    else:
        record_yes = True
        terminal_label.config(text="Recording...")

btn_record = tk.Button(
    root,
    text="Enregistrer",
    command=record_YN,
    font=("Arial", 14)
)
btn_record.pack(pady=5)

# ----------------------------
# Paramètre
# ----------------------------

def save():
    global numero_camera
    global cap
    cap.release()
    cap = cv2.VideoCapture(int(numero_camera.get()))


def type_enregistrement():
    global type_record
    global type_b
    if type_record == "video":
        type_record = "photo"
        type_b.config(text="type d'enregistrement: Photo")
    else:
        type_record = "video"
        type_b.config(text="type d'enregistrement: Vidéo")



def parametre():
    global numero_camera
    global type_b
    fenetre_para = tk.Toplevel(root)
    fenetre_para.title("parametres")
    fenetre_para.geometry("300x200")


    type_b = tk.Button(
        fenetre_para,
        text="type d'enregistrement: Vidéo",
        command=type_enregistrement
    )
    type_b.pack(side="bottom")


    btn2 = tk.Button(
        fenetre_para,
        text="appliquer",
        command=save
    )
    btn2.pack(side="bottom")

    fenetre_entry = tk.Entry(
        fenetre_para,
        textvariable=numero_camera,
        width=10
    )
    fenetre_entry.pack(pady=10)

    label_camera = tk.Label(fenetre_para, text="Numéro de la Caméra")
    label_camera.pack(pady=10, side="left")

btn_para = tk.Button(
    root,
    text="paramètres",
    command=parametre,
    font=("Arial", 14)
)
btn_para.pack(pady=10)

# ----------------------------
# Fermeture propre
# ----------------------------

def fermer():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_quitter = tk.Button(
    root,
    text="Quitter",
    command=fermer,
    font=("Arial", 14)
)
btn_quitter.pack(pady=10)

# ----------------------------
# Lancement
# ----------------------------

update_frame()
root.protocol("WM_DELETE_WINDOW", fermer)
root.mainloop()