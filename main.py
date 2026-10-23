import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import os
from datetime import date
from openpyxl import Workbook
from datetime import datetime
import time
from ultralytics import YOLO

model_cls = YOLO(
    "runs/classify/microorganismes_v1/weights/best.pt"
)

wb = Workbook() #Création classeur
liste_obj = []
liste_id_excel = set()
# ----------------------------
# Variables globales
# ----------------------------
DETECT_ONLY_MOVING = False   # True = disparaît dès que l'objet s'immobilise
                            # False = reste affiché pendant IMMOBILE_TIMEOUT_SEC
global cap
global MULTIPLE_PHOTOS_INTERVAL
MULTIPLE_PHOTOS_INTERVAL = 3
global MULTIPLE_PHOTOS
MULTIPLE_PHOTOS = False
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

global record_dir
record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures/")
os.makedirs(record_dir, exist_ok=True)


# ----------------------------
# ANALYSE
# ----------------------------

ids_ecrits = set()   # IDs déjà écrits pour l'enregistrement en cours

def ecrire_objet(obj_id, type_objet):
    chemin_texte = os.path.join(record_dir, "log.txt")
    with open(chemin_texte, "a", encoding="utf-8") as f:
        f.write(f"ID:{obj_id} {type_objet}\n")





# ----------------------------
# AJOUT MINIMAL VIDÉOS
# ----------------------------


RECORD_FPS = FPS_ESTIMATE
RECORD_SIZE = (200, 200)
MAX_RECORD_FRAMES = FPS_ESTIMATE * 3
VIDEO_ZOOM_FACTOR = 4

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

                #if area < 200:
                    #type_objet = "Petit"
                #elif circularity > 0.8:
                    #type_objet = "Rond"
                #elif area > 15000:
                    #type_objet = "Grand"
                #else:
                    #type_objet = "Non definit ???"

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
                    #obj["type"] = type_objet
                    obj["lost"] = 0
                    
                else:
                    obj_id = next_id
                    crop = frame[y:y+h, x:x+w]

                    results = model_cls.predict(
                        crop,
                        verbose=False
                    )

                    class_id = results[0].probs.top1
                    type_objet = results[0].names[class_id]

                    tracked_objects[obj_id] = {
                        "x": cx, "y": cy,
                        "w": w, "h": h,
                        "area": area,
                        "type": type_objet,
                        "lost": 0,
                        "cote_video": int(max(w, h) * VIDEO_ZOOM_FACTOR)  # fixé une fois pour toujours
                    }

                    next_id += 1

                seen_this_frame.add(obj_id)

                for obj_id in list(tracked_objects.keys()): #Changement dU mode DYNAMIQUE
                    if obj_id not in seen_this_frame:
                        tracked_objects[obj_id]["lost"] += 1

                    timeout = 1 if DETECT_ONLY_MOVING else IMMOBILE_TIMEOUT_FRAMES

                    if tracked_objects[obj_id]["lost"] > timeout:
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

                #AJOUT AU TABLEAU EXCEL
                if obj_id not in liste_id_excel: 
                    liste_obj.append(obj['type'])
                    print(liste_obj)
                    liste_id_excel.add(obj_id)

                # Écriture dans le texte.txt du dossier d'enregistrement
                if record_yes and obj_id not in ids_ecrits:
                    ecrire_objet(obj_id, obj['type'])
                    ids_ecrits.add(obj_id)




                # 🎥 MODE VIDÉO
                if type_record == "video":
        
                    if obj_id not in recording_done:

                        cote = obj["cote_video"]
                        vx0 = max(0, obj["x"] - cote // 2)
                        vy0 = max(0, obj["y"] - cote // 2)
                        vx1 = min(fw, obj["x"] + cote // 2)
                        vy1 = min(fh, obj["y"] + cote // 2)

                        if vx1 > vx0 and vy1 > vy0:

                            crop = frame[vy0:vy1, vx0:vx1]
                            crop_resized = cv2.resize(crop, RECORD_SIZE)

                            recordings.setdefault(obj_id, []).append(crop_resized)

                            if len(recordings[obj_id]) >= MAX_RECORD_FRAMES and record_yes:

                                filename = os.path.join(
                                    record_dir,
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

                    if record_yes:

                        fh2, fw2 = frame.shape[:2]
                        x0 = max(0, x)
                        y0 = max(0, y)
                        x1 = min(fw2, x + w)
                        y1 = min(fh2, y + h)

                        if x1 > x0 and y1 > y0:

                            now = time.time()

                            # Vérifie si on peut prendre une photo pour cet objet
                            last_photo_time = obj.get("last_photo", 0)

                            if MULTIPLE_PHOTOS:
                                # On prend une photo toutes les X secondes
                                peut_prendre = (now - last_photo_time) >= MULTIPLE_PHOTOS_INTERVAL
                            else:
                                # Comportement actuel : 1 seule photo par ID
                                peut_prendre = obj_id not in photo_done

                            if peut_prendre:

                                crop = frame[y0:y1, x0:x1]

                                filename = os.path.join(
                                    record_dir,
                                    f"obj_{obj_id}_{obj['type']}_{int(now*1000)}.png"
                                )

                                ok = cv2.imwrite(filename, crop)

                                if ok:
                                    print(f"[OK] Photo enregistrée : {filename}")
                                    obj["last_photo"] = now      # mémorise l'heure de la dernière photo
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

def folder_name_function(folder_name_num):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    folder_name = str(date.today()) + "_" + str(int(folder_name_num))
    if os.path.isdir(os.path.join(base, folder_name)):
        return folder_name_function(int(folder_name_num) + 1)
    else:
        return folder_name

def record_YN():
    global record_yes
    global record_dir

    if record_yes:
        record_yes = False
        terminal_label.config(text="Stop recording !")
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        record_dir = os.path.join(base, folder_name_function(0))
        os.makedirs(record_dir, exist_ok=True)
        ids_ecrits.clear()

        # Séparateur de session (les ID repartent de 0 à chaque lancement)
        text_path = os.path.join(record_dir, "log.txt")
        with open(text_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Session du {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

        record_yes = True
        terminal_label.config(text="Recording...")
        print(f"[INFO] Les enregistrements seront sauvegardés dans : {record_dir}")



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

def moving():
    global DETECT_ONLY_MOVING
    global type_m
    if DETECT_ONLY_MOVING :
        DETECT_ONLY_MOVING = False
        type_m.config(text="type de detection: Moving & Stop")
        print(DETECT_ONLY_MOVING)
    else:
        DETECT_ONLY_MOVING = True
        type_m.config(text="type de detection: Only Moving")
        print(DETECT_ONLY_MOVING)

def multiple():
    global MULTIPLE_PHOTOS
    global type_mp
    if MULTIPLE_PHOTOS :
        MULTIPLE_PHOTOS = False
        type_mp.config(text="Single photo")
        print(MULTIPLE_PHOTOS)
    else:
        MULTIPLE_PHOTOS = True
        type_mp.config(text="multiple photos")
        print(MULTIPLE_PHOTOS)


    
def parametre():
    global numero_camera
    global MULTIPLE_PHOTOS
    global type_b
    global type_m
    global type_mp
    global DETECT_ONLY_MOVING
    fenetre_para = tk.Toplevel(root)
    fenetre_para.title("parametres")
    fenetre_para.geometry("300x200")

    if DETECT_ONLY_MOVING:
        DETECT_ONLY_MOVING_TEXT = "type de detection: Only Moving"
    else:
        DETECT_ONLY_MOVING_TEXT = "type de detection: Moving & Stop"

    if MULTIPLE_PHOTOS:
        MULTIPLE_PHOTOS_TEXT = "Single photo"
    else:
        MULTIPLE_PHOTOS_TEXT = "multiple photos"


    print(DETECT_ONLY_MOVING)



    type_m = tk.Button(
        fenetre_para,
        text= DETECT_ONLY_MOVING_TEXT,
        command=moving
    )
    type_m.pack(side="bottom")

    type_mp = tk.Button(
        fenetre_para,
        text=MULTIPLE_PHOTOS_TEXT,
        command=multiple
    )
    type_mp.pack(side="bottom")


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


#######################
#FENETRE ANALYSES######
#######################

def analyse_window():
    #global 
    fenetre_analyse = tk.Toplevel(root)
    fenetre_analyse.title("Analyse")
    fenetre_analyse.geometry("300x200")


    type_m = tk.Button(
        fenetre_analyse,
        text= DETECT_ONLY_MOVING_TEXT,
        command=moving
    )
    type_m.pack(side="bottom")

    type_mp = tk.Button(
        fenetre_analyse,
        text=MULTIPLE_PHOTOS_TEXT,
        command=multiple
    )
    type_mp.pack(side="bottom")


    type_b = tk.Button(
        fenetre_analyse,
        text="type d'enregistrement: Vidéo",
        command=type_enregistrement
    )
    type_b.pack(side="bottom")


    btn2 = tk.Button(
        fenetre_analyse,
        text="appliquer",
        command=save
    )
    btn2.pack(side="bottom")

    fenetre_entry = tk.Entry(
        fenetre_analyse,
        textvariable=numero_camera,
        width=10
    )
    fenetre_entry.pack(pady=10)

    label_camera = tk.Label(fenetre_analyse, text="Numéro de la Caméra")
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