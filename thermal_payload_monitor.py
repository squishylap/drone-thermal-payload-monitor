"""
Drone Thermal Payload Monitoring & Alerting Engine
----------------------------------------------------
Goal: Detect overheating payload sections in real time using
thermal image data.

This script:
1. Loads/simulates a sequence of thermal frames
2. Extracts thermal gradients (edge detection) using NumPy + OpenCV
3. Isolates hotspots using a threshold-based binary mask
4. Tracks hotspots over time and logs overheat alerts

Author: Muhammad Hassan
Course project - ME (First Year)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------
# STEP 1: Simulate a thermal frame
# ---------------------------------------------------------
# In a real drone system, this would come from an FLIR/UAV thermal
# sensor. Since we don't have a live sensor feed, we simulate a
# single thermal frame: a noisy background with one "hot" circular
# region representing an overheating payload section.

def generate_thermal_frame(width=640, height=480, hotspot_center=(200, 160), hotspot_radius=25):
    # background noise (simulates ambient thermal readings)
    frame = np.random.normal(loc=40, scale=5, size=(height, width))

    # add a hot circular region (simulates overheating payload part)
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - hotspot_center[0])**2 + (y - hotspot_center[1])**2)
    hotspot_mask = distance_from_center <= hotspot_radius
    frame[hotspot_mask] = np.random.normal(loc=95, scale=3, size=frame[hotspot_mask].shape)

    return frame


# ---------------------------------------------------------
# STEP 2: Extract thermal gradients (edge detection)
# ---------------------------------------------------------
# We use OpenCV's Sobel operator to find where temperature changes
# sharply - this highlights the edges of the hot region.

def extract_gradients(frame):
    frame_uint8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grad_x = cv2.Sobel(frame_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(frame_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    return gradient_magnitude


# ---------------------------------------------------------
# STEP 3: Isolate hotspots using a threshold
# ---------------------------------------------------------
# Any pixel above the temperature threshold is marked as "hot".
# This gives us a binary mask: white = hotspot, black = normal.

def isolate_hotspots(frame, threshold=80):
    hotspot_mask = np.where(frame > threshold, 255, 0).astype(np.uint8)
    return hotspot_mask


# ---------------------------------------------------------
# STEP 4: Generate alert if overheating is detected
# ---------------------------------------------------------
# If any pixel crosses the critical threshold, log an alert with
# a timestamp and the approximate location of the hotspot.

def check_for_alert(frame, hotspot_mask, critical_threshold=90):
    alerts = []

    if np.any(frame > critical_threshold):
        ys, xs = np.where(hotspot_mask == 255)
        if len(xs) > 0 and len(ys) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            timestamp = datetime.now().strftime("%H:%M:%S")
            alerts.append(f"[{timestamp}] ALERT: Overheat at ({center_x}, {center_y})")

    return alerts


# ---------------------------------------------------------
# STEP 5: Run the full pipeline and display results
# ---------------------------------------------------------

def main():
    print("Generating thermal frame...")
    thermal_frame = generate_thermal_frame()

    print("Extracting thermal gradients...")
    gradients = extract_gradients(thermal_frame)

    print("Isolating hotspots...")
    hotspot_mask = isolate_hotspots(thermal_frame, threshold=80)

    print("Checking for overheat alerts...")
    alert_log = check_for_alert(thermal_frame, hotspot_mask, critical_threshold=90)

    if alert_log:
        print("SYSTEM STATUS: OVERHEAT ALERT")
        for alert in alert_log:
            print(alert)
    else:
        print("SYSTEM STATUS: Normal")

    # -------------------------------------------------
    # Display the 4-panel output (matches project report)
    # -------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(thermal_frame, cmap='inferno')
    axs[0, 0].set_title("1. Raw Thermal Feed (Inferno Map)")

    axs[0, 1].imshow(gradients, cmap='gray')
    axs[0, 1].set_title("2. Thermal Gradients (Edge Detection)")

    axs[1, 0].imshow(hotspot_mask, cmap='gray')
    axs[1, 0].set_title("3. Hotspot Isolation (Binary Mask)")

    axs[1, 1].axis('off')
    axs[1, 1].set_title("4. THERMAL PAYLOAD ALERT PANEL", color='red')
    alert_text = "\n".join(alert_log) if alert_log else "No alerts triggered."
    axs[1, 1].text(0.05, 0.7, alert_text, fontsize=10, color='red', wrap=True)

    plt.tight_layout()
    plt.savefig("thermal_payload_alert_panel.png", dpi=150)
    plt.show()
    print("\nOutput saved as 'thermal_payload_alert_panel.png'")


if __name__ == "__main__":
    main()
