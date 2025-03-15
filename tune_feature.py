import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import pandas as pd
import os

# Load pre-trained ResNet50 for deep feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
deep_feature_model = Model(inputs=resnet_model.input, outputs=resnet_model.output)

# Function to detect circles based on parameters
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
    return circles

# Function to calculate deep features using ResNet50
def calculate_features(image, circles):
    features = []
    height, width = image.shape

    for circle in circles:
        x, y, radius = circle

        # Validate and calculate the bounding box
        x1, y1 = max(0, int(x - radius)), max(0, int(y - radius))
        x2, y2 = min(width, int(x + radius)), min(height, int(y + radius))

        # Check for invalid ROI dimensions
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid ROI for circle at X={x}, Y={y}, Radius={radius}. Skipping...")
            continue

        # Extract the ROI
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
        roi = cv2.bitwise_and(image, image, mask=mask)[y1:y2, x1:x2]

        # Ensure ROI is non-empty
        if roi.size == 0:
            print(f"Empty ROI for circle at X={x}, Y={y}, Radius={radius}. Skipping...")
            continue

        # Resize and preprocess for ResNet50
        resized_roi = cv2.resize(roi, (224, 224))
        resized_roi_rgb = cv2.cvtColor(resized_roi, cv2.COLOR_GRAY2RGB)
        normalized_roi = preprocess_input(np.expand_dims(resized_roi_rgb, axis=0))

        # Extract deep features
        deep_features = deep_feature_model.predict(normalized_roi).flatten()

        # Save features
        feature_dict = {"X": x, "Y": y, "Radius": radius}
        for k, feature in enumerate(deep_features, 1):
            feature_dict[f"Deep_Feature_{k}"] = feature
        features.append(feature_dict)

    return features

# Update the visualization when slider values change
def update(val):
    global detected_circles
    dp = dp_slider.val
    minDist = minDist_slider.val
    param1 = param1_slider.val
    param2 = param2_slider.val
    minRadius = minRadius_slider.val
    maxRadius = maxRadius_slider.val

    # Detect circles with current slider values
    detected_circles = detect_circles(
        image, dp, minDist, param1, param2, int(minRadius), int(maxRadius)
    )
    
    ax.clear()
    ax.imshow(image, cmap="gray")
    ax.set_title("Tune the Parameters Using Sliders Below")
    
    # Draw detected circles
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            x, y, radius = circle
            circ = plt.Circle((x, y), radius, color="green", fill=False, linewidth=1.5)
            ax.add_artist(circ)

    fig.canvas.draw_idle()

# Save features for all detected circles
def save_features(event):
    global detected_circles
    print("Save Features button clicked.")
    
    if detected_circles is None or len(detected_circles[0]) == 0:
        print("No circles detected.")
        return
    
    print(f"Detected {len(detected_circles[0])} circles. Calculating features...")
    features = calculate_features(image, detected_circles[0])
    
    if features:
        features_df = pd.DataFrame(features)
        # Save the CSV file in the same folder as the input image
        image_folder = os.path.dirname(image_path)  # Get the folder of the input image
        output_path = os.path.join(image_folder, "all_detected_circle_features.csv")
        
        try:
            features_df.to_csv(output_path, index=False)
            print(f"Features for all detected circles saved to: {output_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("No features calculated. Ensure valid circle detection.")

# Load the grayscale image
image_path = r"C:\\vijay\\Python\\Image_analysis\\Images\\automated classification\\Detect circles\\Images\\Model training\\white image\\data1\\tune_feature\\1.tif"

if not os.path.exists(image_path):
    raise ValueError(f"File not found: {image_path}. Please check the file path.")

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"OpenCV could not read the file: {image_path}. Try converting the file to a supported format or check the file integrity.")

# Default parameter values
default_dp = 1.2
default_minDist = 20
default_param1 = 100
default_param2 = 30
default_minRadius = 5
default_maxRadius = 25

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.4)
ax.imshow(image, cmap="gray")
ax.set_title("Tune the Parameters Using Sliders Below")

# Add sliders below the image
axcolor = "lightgoldenrodyellow"
ax_dp = plt.axes([0.2, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_minDist = plt.axes([0.2, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_param1 = plt.axes([0.2, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_param2 = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_minRadius = plt.axes([0.2, 0.10, 0.30, 0.03], facecolor=axcolor)
ax_maxRadius = plt.axes([0.6, 0.10, 0.30, 0.03], facecolor=axcolor)

dp_slider = Slider(ax_dp, "dp", 0.5, 3.0, valinit=default_dp, valstep=0.1)
minDist_slider = Slider(ax_minDist, "minDist", 10, 100, valinit=default_minDist, valstep=1)
param1_slider = Slider(ax_param1, "param1", 50, 250, valinit=default_param1, valstep=1)
param2_slider = Slider(ax_param2, "param2", 10, 100, valinit=default_param2, valstep=1)
minRadius_slider = Slider(ax_minRadius, "minRadius", 0, 80, valinit=default_minRadius, valstep=1)
maxRadius_slider = Slider(ax_maxRadius, "maxRadius", 10, 100, valinit=default_maxRadius, valstep=1)

# Add Save Features button
ax_save = plt.axes([0.8, 0.01, 0.1, 0.05])
save_button = Button(ax_save, "Save Features", color=axcolor, hovercolor="0.975")
save_button.on_clicked(save_features)

# Connect sliders
dp_slider.on_changed(update)
minDist_slider.on_changed(update)
param1_slider.on_changed(update)
param2_slider.on_changed(update)
minRadius_slider.on_changed(update)
maxRadius_slider.on_changed(update)

# Initial update
update(None)

plt.show()
