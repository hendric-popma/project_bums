import cv2
import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def process_frame(frame):
    # Hier kannst du den Frame verarbeiten, z.B. Bildverarbeitungsalgorithmen anwenden
    # ...

    # Text in Audio umwandeln und abspielen
    text_to_speech("Dies ist ein Beispieltext.")

    # Hier kannst du weitere Operationen auf dem Frame durchführen, falls erforderlich
    # ...

    return frame

def process_video(video_file):
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = process_frame(frame)

        # Hier kannst du den verarbeiteten Frame anzeigen, speichern oder anderweitig verwenden
        cv2.imshow('Processed Frame', processed_frame)
        
        # Warte auf das Drücken der Taste 'q' zum Beenden der Schleife
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


process_video("C:\\Users\\leonh\\Pictures\\BumS\\Projekt\\Videos\\muc_bhf_3.mp4")
