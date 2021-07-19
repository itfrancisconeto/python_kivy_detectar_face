import cv2

class Detectar_face(object):
    
    def detectarFaceEmCam(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            for (x, y, w, h) in faces:
                faceROI = gray[y:y+h,x:x+w]
                smile = smile_detector.detectMultiScale(
                    faceROI, 
                    3.5, 
                    5
                )
                for (x3,y3,w3,h3) in smile:
                    cv2.rectangle(frame,(x+x3,y+y3),(x+x3+w3,y+y3+h3),(0,0,255),3)
                    print("ESTA SORRINDO!")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':    
    detectarFace = Detectar_face()
    detectarFace.detectarFaceEmCam()
