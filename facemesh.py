import cv2 as cv
import time
import mediapipe as mp


class facemesh:
    def __init__(self,static_image_mode=False,max_num_faces=2,refine_landmarks=False,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_tracking_confidence
        self.min_tracking_confidence = min_tracking_confidence


        # defining modules
        self.mpfacemesh = mp.solutions.face_mesh
        self.facemesh = self.mpfacemesh.FaceMesh(self.static_image_mode, self.max_num_faces, self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    #-------------------
    def findfacemesh(self,frame,draw=False):
        RGBimg = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.facemesh.process(RGBimg)


        lst  = []
        if self.result.multi_face_landmarks:
            for lm in self.result.multi_face_landmarks:
                self.mpDraw.draw_landmarks(frame,lm,self.mpfacemesh.FACE_CONNECTIONS,self.mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(0,255,0)))

                for id,getlm in enumerate(lm.landmark):
                    print(getlm)
                    ih,iw,ic= frame.shape
                    x,y = int(getlm.x*iw),(getlm.y*ih)
                    print(x,y)
        return frame



# ------------------------end class

def main():
    cap = cv.VideoCapture('v2.mp4')
    detector = facemesh()
    cTime = 0
    pTime = 0
    while True:
        success,frame = cap.read()
        frame = detector.findfacemesh(frame)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(frame,f"FPS: {int(fps)})",(15,30),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),3)
        cv.imshow('image',frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    cv.release()
    cv.destroyAllWindows()

# ----------------python main
if __name__ == "__main__":
    main()