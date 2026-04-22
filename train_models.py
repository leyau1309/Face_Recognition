import cv2, os, numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset_path = r"C:\Users\User\Downloads\faceRecAtt\train" 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

X, y, label_map = [], [], {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    label_map[current_label] = person_name
    for img_name in os.listdir(person_path)[:20]:
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x1, y1, w, h) in faces_rect:
            face = gray[y1:y1+h, x1:x1+w]
            face = cv2.resize(face, (100,100))
            X.append(face)
            y.append(current_label)
    current_label += 1

X = np.array(X)
y = np.array(y)

# --- Eigenfaces (PCA + SVM) ---
X_flat = X.reshape(len(X), -1)
pca = PCA(n_components=20)  # smaller for speed
X_pca = pca.fit_transform(X_flat)
svm_model = SVC(kernel='linear')
svm_model.fit(X_pca, y)

# --- KNN ---
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_flat, y)

# --- LBPH ---
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.train(list(X), list(y))

# --- Save models ---
with open("pca.pkl", "wb") as f: pickle.dump(pca, f)
with open("svm.pkl", "wb") as f: pickle.dump(svm_model, f)
with open("knn.pkl", "wb") as f: pickle.dump(knn_model, f)
lbph_model.save("lbph.yml")
with open("label_map.pkl", "wb") as f: pickle.dump(label_map, f)

print("Training completed! Models saved.")