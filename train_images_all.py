import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Load pre-trained ResNet50 for deep feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
deep_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.output)

# Manual GLCM computation
def compute_glcm(image, distances, angles, levels=256):
    rows, cols = image.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32)

    for d_idx, distance in enumerate(distances):
        for a_idx, angle in enumerate(angles):
            dx = int(np.round(distance * np.cos(angle)))
            dy = int(np.round(distance * np.sin(angle)))

            for i in range(rows):
                for j in range(cols):
                    if 0 <= i + dy < rows and 0 <= j + dx < cols:
                        ref_pixel = image[i, j]
                        neighbor_pixel = image[i + dy, j + dx]
                        glcm[ref_pixel, neighbor_pixel, d_idx, a_idx] += 1
    return glcm

def compute_glcm_properties(glcm):
    properties = {}
    glcm_sum = np.sum(glcm)
    if glcm_sum == 0:
        return {"contrast": 0, "energy": 0, "homogeneity": 0, "correlation": 0}

    normalized_glcm = glcm / glcm_sum

    # Contrast
    contrast = np.sum([(i - j) ** 2 * normalized_glcm[i, j] for i in range(normalized_glcm.shape[0]) for j in range(normalized_glcm.shape[1])])
    properties["contrast"] = contrast

    # Energy
    properties["energy"] = np.sum(normalized_glcm**2)

    # Homogeneity
    homogeneity = np.sum([normalized_glcm[i, j] / (1 + abs(i - j)) for i in range(normalized_glcm.shape[0]) for j in range(normalized_glcm.shape[1])])
    properties["homogeneity"] = homogeneity

    # Correlation
    i_indices, j_indices = np.meshgrid(range(normalized_glcm.shape[0]), range(normalized_glcm.shape[1]), indexing="ij")
    mean_i = np.sum(i_indices * normalized_glcm)
    mean_j = np.sum(j_indices * normalized_glcm)
    std_i = np.sqrt(np.sum((i_indices - mean_i)**2 * normalized_glcm))
    std_j = np.sqrt(np.sum((j_indices - mean_j)**2 * normalized_glcm))
    correlation = np.sum((i_indices - mean_i) * (j_indices - mean_j) * normalized_glcm) / (std_i * std_j)
    properties["correlation"] = correlation

    return properties

def detect_circles(image, dp, minDist, param1, param2, minRadius, maxRadius):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )
    return None if circles is None else np.uint16(np.around(circles[0]))

def calculate_features(image, circles):
    features = []
    for circle in circles:
        x, y, radius = circle
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(image.shape[1], x + radius), min(image.shape[0], y + radius)
        cropped_circle = roi[y1:y2, x1:x2]

        # Intensity features
        mean_intensity = cv2.mean(image, mask=mask)[0]
        integrated_intensity = mean_intensity * (np.pi * radius**2)

        # Texture features using manual GLCM
        glcm = compute_glcm(cropped_circle, distances=[1], angles=[0])
        glcm_props = compute_glcm_properties(glcm[:, :, 0, 0])

        # Deep features using ResNet50
        resized_roi = cv2.resize(cropped_circle, (224, 224))
        resized_roi_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_GRAY2RGB)
        normalized_roi = preprocess_input(np.expand_dims(resized_roi_rgb, axis=0))
        deep_features = deep_feature_model.predict(normalized_roi).flatten()

        feature_dict = {
            "X": x,
            "Y": y,
            "Radius": radius,
            "Mean_Intensity": mean_intensity,
            "Integrated_Intensity": integrated_intensity,
            "Contrast": glcm_props["contrast"],
            "Homogeneity": glcm_props["homogeneity"],
            "Energy": glcm_props["energy"],
            "Correlation": glcm_props["correlation"],
        }
        for k, df in enumerate(deep_features, 1):
            feature_dict[f"Deep_Feature_{k}"] = df
        features.append(feature_dict)
    return features

# Interactive circle selection
def process_image(image, filename, output_folder):
    detected_circles = detect_circles(image, dp=1.2, minDist=80, param1=100, param2=30, minRadius=5, maxRadius=25)
    selected_circles = []

    def on_click(event):
        if detected_circles is not None:
            for circle in detected_circles:
                x, y, radius = circle
                distance = np.sqrt((x - event.xdata) ** 2 + (y - event.ydata) ** 2)
                if distance <= radius:
                    selected_circles.append(circle)
                    print(f"Selected Circle: X={x}, Y={y}, Radius={radius}")
                    break

    def save_selected_features(event):
        if not selected_circles:
            print("No circles selected.")
            return

        features = calculate_features(image, selected_circles)
        if features:
            features_df = pd.DataFrame(features)
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_features.csv")
            features_df.to_csv(output_file, index=False)
            print(f"Features for selected circles saved to {output_file}")
        else:
            print("No features to save. Ensure selected circles are valid.")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap="gray")
    if detected_circles is not None:
        for circle in detected_circles:
            x, y, radius = circle
            ax.add_artist(plt.Circle((x, y), radius, color="green", fill=False, linewidth=1.5))

    ax_save = plt.axes([0.8, 0.01, 0.1, 0.05])
    save_button = Button(ax_save, "Save Features")
    save_button.on_clicked(save_selected_features)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

def process_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            print(f"Processing {filename}")
            process_image(image, filename, output_folder)

# Specify input and output directories
input_folder = "C:\\vijay\\Python\\Image_analysis\\Images\\automated classification\\Detect circles\\Images\\Model training\\white image\\data1"
output_folder = "C:\\vijay\\Python\\Image_analysis\\Images\\automated classification\\Detect circles\\Images\\Model training\\white image\\data1\\feature extraction"

process_folder(input_folder, output_folder)
