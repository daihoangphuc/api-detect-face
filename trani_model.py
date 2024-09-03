import os
import face_recognition
import numpy as np
import joblib
from sklearn import neighbors
from rich.progress import Progress

def load_images_from_folder(folder):
    images = []
    labels = []
    print("Loading images from folder...")
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        if os.path.isdir(subfolder):
            file_list = os.listdir(subfolder)
            print(f"Found {len(file_list)} files in folder '{subdir}'")
            with Progress() as progress:
                task = progress.add_task(f"[green]Processing {subdir}...", total=len(file_list))
                for filename in file_list:
                    img_path = os.path.join(subfolder, filename)
                    img = face_recognition.load_image_file(img_path)
                    img_encoding = face_recognition.face_encodings(img)
                    if img_encoding:
                        images.append(img_encoding[0])
                        labels.append(subdir)  # Đảm bảo nhãn chứa tên thư mục đầy đủ
                    progress.update(task, advance=1)  # Cập nhật thanh tiến độ
    print(f"Loaded {len(images)} images.")
    return images, labels

# Đường dẫn đến thư mục chứa ảnh
folder_path = 'D:/@training_face/Deploy_API_Heruku/training_face'
images, labels = load_images_from_folder(folder_path)

# Huấn luyện bộ phân loại KNN
print("Training KNN classifier...")
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', weights='distance')
knn_clf.fit(images, labels)
print("Training completed.")

# Lưu mô hình đã huấn luyện
model_path = 'Deploy_API_Heruku/knn_model.pkl'
print(f"Saving the trained model to {model_path}...")
joblib.dump(knn_clf, model_path)
print("Model saved successfully.")
