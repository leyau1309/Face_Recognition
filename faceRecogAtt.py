import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from collections import Counter

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Face Recognition System", layout="wide")

dataset_path = "train"
attendance_file = "attendance.csv"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# FIX 3: CLAHE only — no double histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# -----------------------------
# QUALITY CHECK
# -----------------------------
# FIX 2: Lowered blur threshold from 40 → 15 to keep more usable samples
def is_blurry(img, threshold=15):
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold

def preprocess(face):
    face = cv2.resize(face, (100, 100))
    # FIX 3: CLAHE only — removed equalizeHist to avoid over-processing
    face = clahe.apply(face)
    return face

# -----------------------------
# ATTENDANCE
# -----------------------------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if not os.path.exists(attendance_file):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

    df = pd.read_csv(attendance_file)

    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_row = pd.DataFrame([[name, date, time_str]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return True
    return False

def get_attendance():
    if not os.path.exists(attendance_file):
        return pd.DataFrame(columns=["Name", "Date", "Time"])
    return pd.read_csv(attendance_file)

# -----------------------------
# LOAD DATASET
# FIX 1: Removed @st.cache_resource so retraining works properly.
#         Use the manual "Retrain" button in the UI instead.
# -----------------------------
def load_data():
    X_list, y_list = [], []
    label_map = {}
    label = 0

    people = [p for p in os.listdir(dataset_path)
              if os.path.isdir(os.path.join(dataset_path, p))]

    if len(people) == 0:
        return None, None, None

    for person in sorted(people):
        path = os.path.join(dataset_path, person)
        label_map[label] = person
        count = 0

        for img_name in os.listdir(path):
            if count >= 100:   # Increased from 50 to allow more samples
                break
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # FIX 6: Looser detection — scaleFactor=1.1, minNeighbors=4
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                if is_blurry(face):
                    continue
                face = preprocess(face)
                X_list.append(face)
                y_list.append(label)
                count += 1

        label += 1

    if len(X_list) == 0:
        return None, None, None

    X = np.array(X_list)
    y = np.array(y_list)
    X_flat = X.reshape(len(X), -1) / 255.0

    return X_flat, y, label_map

# -----------------------------
# TRAIN ALL MODELS
# -----------------------------
def train_models():
    X, y, label_map = load_data()

    if X is None:
        return None

    # Need at least 2 samples per class for train/test split
    unique, counts = np.unique(y, return_counts=True)
    valid_labels = unique[counts >= 2]
    mask = np.isin(y, valid_labels)
    X, y = X[mask], y[mask]

    if len(X) < 4:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Eigenfaces (PCA + SVM)
    n_components = min(50, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_components, whiten=True)  # whiten=True improves SVM
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)  # RBF better than linear
    svm.fit(X_train_pca, y_train)

    # FIX 4: KNN now uses PCA features instead of raw pixels
    knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train) // len(np.unique(y_train))))
    knn.fit(X_train_pca, y_train)

    # LBPH
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    lbph.train(
        [x.reshape(100, 100).astype(np.uint8) for x in X_train],
        np.array(y_train)
    )

    return {
        "pca": pca,
        "svm": svm,
        "knn": knn,
        "lbph": lbph,
        "label_map": label_map,
        "X_test": X_test,
        "X_test_pca": X_test_pca,
        "y_test": y_test,
    }

# -----------------------------
# SESSION STATE — train once, retrain on button press
# -----------------------------
if "models" not in st.session_state:
    with st.spinner("Training models..."):
        st.session_state.models = train_models()

models = st.session_state.models

if models is None:
    st.warning("⚠️ No training data found. Please register at least one person first.")
    models = {}  # Allow UI to still render

# -----------------------------
# METRICS
# -----------------------------
def evaluate(model_name):
    if not models:
        return {}

    pca       = models["pca"]
    svm       = models["svm"]
    knn       = models["knn"]
    lbph      = models["lbph"]
    X_test    = models["X_test"]
    X_test_pca= models["X_test_pca"]
    y_test    = models["y_test"]

    if model_name == "Eigenfaces":
        pred = svm.predict(X_test_pca)
    elif model_name == "KNN":
        # FIX 4: KNN predicts on PCA features
        pred = knn.predict(X_test_pca)
    else:  # LBPH
        pred = np.array([
            lbph.predict(x.reshape(100, 100).astype(np.uint8))[0]
            for x in X_test
        ])

    avg = "macro" if len(np.unique(y_test)) > 2 else "binary"
    return {
        "Accuracy":  round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred, average=avg, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, pred, average=avg, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_test, pred, average=avg, zero_division=0), 4),
    }

# -----------------------------
# RECOGNITION
# FIX 5: LBPH confidence gate — reject if confidence > 80
# BONUS: Majority vote across all 3 algorithms
# -----------------------------
LBPH_CONFIDENCE_THRESHOLD = 80

def recognize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # FIX 6: Looser detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    results = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        if is_blurry(face_roi):
            continue

        face_proc = preprocess(face_roi)
        face_flat = face_proc.reshape(1, -1) / 255.0
        face_pca  = models["pca"].transform(face_flat)

        # Eigenfaces prediction
        eigen_label = models["svm"].predict(face_pca)[0]
        eigen_name  = models["label_map"][eigen_label]

        # KNN prediction (FIX 4: on PCA space)
        knn_label = models["knn"].predict(face_pca)[0]
        knn_name  = models["label_map"][knn_label]

        # LBPH prediction with confidence gate (FIX 5)
        lbph_label, lbph_conf = models["lbph"].predict(face_proc)
        lbph_name = models["label_map"][lbph_label] if lbph_conf <= LBPH_CONFIDENCE_THRESHOLD else "Unknown"

        # BONUS: Majority vote
        votes = [eigen_name, knn_name, lbph_name]
        vote_count = Counter(v for v in votes if v != "Unknown")
        if vote_count:
            majority_name = vote_count.most_common(1)[0][0]
        else:
            majority_name = "Unknown"

        results.append({
            "Eigenfaces": eigen_name,
            "KNN":        knn_name,
            "LBPH":       lbph_name,
            "Majority Vote": majority_name,
            "LBPH Confidence": round(lbph_conf, 1),
        })

        color = (0, 255, 0) if majority_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, majority_name, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, results

# -----------------------------
# UI
# -----------------------------
st.title("🚀 Face Recognition Attendance System")

tab1, tab2, tab3, tab4 = st.tabs(["Recognition", "Register", "Evaluation", "Attendance"])

# -----------------------------
# TAB 1 — RECOGNITION
# -----------------------------
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        algo = st.selectbox(
            "Algorithm for attendance marking",
            ["Majority Vote", "Eigenfaces", "KNN", "LBPH"],
            help="Majority Vote uses all 3 algorithms together for best accuracy"
        )
        mode = st.radio("Input mode", ["Webcam", "Upload Image"])

        file = None
        if mode == "Webcam":
            file = st.camera_input("Capture face")
        else:
            file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    with col_right:
        st.markdown("**Algorithm guide**")
        st.markdown("- **Majority Vote** — most accurate, uses all 3")
        st.markdown("- **Eigenfaces** — PCA + SVM, good overall")
        st.markdown("- **KNN** — simple, works well with enough data")
        st.markdown("- **LBPH** — robust to lighting changes")

    if file and models:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        result_img, results = recognize(img)
        st.image(result_img, channels="BGR", caption="Detection result")

        if not results:
            st.warning("No face detected or face was too blurry. Try better lighting or move closer.")
        else:
            for r in results:
                st.markdown("---")
                cols = st.columns(4)
                cols[0].metric("Eigenfaces", r["Eigenfaces"])
                cols[1].metric("KNN", r["KNN"])
                cols[2].metric("LBPH", r["LBPH"])
                cols[3].metric("Majority Vote", r["Majority Vote"])
                st.caption(f"LBPH confidence score: {r['LBPH Confidence']} (lower = better match)")

                name = r[algo]
                if name != "Unknown":
                    if st.button(f"✅ Confirm attendance: {name}", key=name):
                        if mark_attendance(name):
                            st.success(f"Attendance marked for {name}")
                        else:
                            st.info(f"{name} already marked today")
                else:
                    st.error("Face not recognized. Please try again or register this person.")

# -----------------------------
# TAB 2 — REGISTER
# -----------------------------
with tab2:
    st.subheader("Register new person")
    st.info("Upload at least 10–20 clear face photos for best results. "
            "Use varied lighting and angles. Click **Retrain** after saving.")

    name = st.text_input("Full name")
    files = st.file_uploader(
        "Upload face images (10–20 recommended)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save images"):
            if name and files:
                path = os.path.join(dataset_path, name)
                os.makedirs(path, exist_ok=True)
                saved = 0

                for i, f in enumerate(files):
                    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # FIX 6: Looser detection for registration too
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        if is_blurry(face):
                            continue
                        face = preprocess(face)
                        cv2.imwrite(f"{path}/{i}.jpg", face)
                        saved += 1
                        break  # one face per image

                if saved > 0:
                    st.success(f"Saved {saved} face(s) for **{name}**. Click **Retrain models** now.")
                else:
                    st.error("No usable faces found. Check image quality and lighting.")
            else:
                st.warning("Enter a name and upload at least one image.")

    with col2:
        if st.button("🔄 Retrain models"):
            with st.spinner("Retraining all models..."):
                st.session_state.models = train_models()
                models = st.session_state.models
            if models:
                st.success("Models retrained successfully!")
            else:
                st.error("Training failed. Need at least 2 people registered.")

# -----------------------------
# TAB 3 — EVALUATION
# -----------------------------
with tab3:
    st.subheader("Model evaluation metrics")

    if not models:
        st.warning("No trained models available.")
    else:
        cols = st.columns(3)
        for i, m in enumerate(["Eigenfaces", "KNN", "LBPH"]):
            metrics = evaluate(m)
            with cols[i]:
                st.markdown(f"**{m}**")
                if metrics:
                    for k, v in metrics.items():
                        st.metric(k, f"{v:.1%}")
                else:
                    st.write("Not enough data")

        st.markdown("---")
        st.markdown("**Interpretation guide**")
        st.markdown(
            "- Accuracy above 85% is good for 3+ people with 10+ images each.\n"
            "- Low recall usually means too few training images.\n"
            "- Low precision means the model is confusing people — add more varied photos.\n"
            "- LBPH confidence < 50 = strong match; > 80 = rejected as unknown."
        )

# -----------------------------
# TAB 4 — ATTENDANCE
# -----------------------------
with tab4:
    st.subheader("Attendance records")

    df = get_attendance()
    if df.empty:
        st.info("No attendance records yet.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            today = datetime.now().strftime("%Y-%m-%d")
            today_count = len(df[df["Date"] == today]) if "Date" in df.columns else 0
            st.metric("Today's attendance", today_count)
            st.metric("Total records", len(df))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")