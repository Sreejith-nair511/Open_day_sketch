import cv2
import numpy as np
import pyautogui
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Game control parameters
prev_action = None
action_cooldown = 0.8  # Cooldown in seconds
last_action_time = time.time()

# Define initial threshold values for HSV color space (for skin/object detection)
# You'll calibrate these values during runtime
lower_color = np.array([0, 50, 50], dtype=np.uint8)
upper_color = np.array([10, 255, 255], dtype=np.uint8)

def define_control_regions(frame_height, frame_width):
    """Define the regions for different game controls"""
    jump_threshold = int(frame_height * 0.3)
    duck_threshold = int(frame_height * 0.7)
    left_threshold = int(frame_width * 0.35)
    right_threshold = int(frame_width * 0.65)
    
    return jump_threshold, duck_threshold, left_threshold, right_threshold

def find_largest_contour(mask):
    """Find the largest contour in the mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    return max(contours, key=cv2.contourArea)

def get_contour_center(contour):
    """Calculate the center point of a contour"""
    if contour is None or cv2.contourArea(contour) < 1000:  # Minimum area threshold
        return None
    
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    return (cx, cy)

def determine_action(center, thresholds):
    """Determine action based on position of the tracked object"""
    if center is None:
        return "NONE"
    
    x, y = center
    jump_threshold, duck_threshold, left_threshold, right_threshold = thresholds
    
    if y < jump_threshold:
        return "JUMP"
    elif y > duck_threshold:
        return "DUCK"
    elif x < left_threshold:
        return "LEFT"
    elif x > right_threshold:
        return "RIGHT"
    else:
        return "NEUTRAL"

def execute_game_action(action):
    """Execute keyboard presses based on detected action"""
    if action == "JUMP":
        pyautogui.press('up')
        return "Jump"
    elif action == "DUCK":
        pyautogui.press('down')
        return "Duck"
    elif action == "LEFT":
        pyautogui.press('left')
        return "Left"
    elif action == "RIGHT":
        pyautogui.press('right')
        return "Right"
    return None

def main():
    global lower_color, upper_color, last_action_time
    
    print("=== Subway Surfers Color Tracking Control ===")
    print("Starting webcam... Please wait.")
    print("\nControls:")
    print("- Press 'c' to enter color calibration mode")
    print("- In calibration mode, press 's' to sample color")
    print("- Press 'q' to quit")
    print("\nGameplay:")
    print("- Move tracked object UP to JUMP")
    print("- Move tracked object DOWN to DUCK")
    print("- Move tracked object LEFT to move LEFT")
    print("- Move tracked object RIGHT to move RIGHT")
    
    # Wait for camera to initialize
    time.sleep(2)
    
    # Variable to track if we're in calibration mode
    calibration_mode = True
    print("\nStarting in calibration mode. Place your colored object in the center box.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Flip horizontally for more intuitive movement
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Define control regions
        thresholds = define_control_regions(height, width)
        jump_threshold, duck_threshold, left_threshold, right_threshold = thresholds
        
        # Draw control region lines
        cv2.line(frame, (0, jump_threshold), (width, jump_threshold), (0, 255, 0), 2)
        cv2.line(frame, (0, duck_threshold), (width, duck_threshold), (0, 0, 255), 2)
        cv2.line(frame, (left_threshold, 0), (left_threshold, height), (255, 0, 0), 2)
        cv2.line(frame, (right_threshold, 0), (right_threshold, height), (255, 0, 0), 2)
        
        # Add labels
        cv2.putText(frame, "JUMP", (10, jump_threshold - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "DUCK", (10, duck_threshold + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "LEFT", (left_threshold - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "RIGHT", (right_threshold + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if calibration_mode:
            # Draw a rectangle in the center for color sampling
            cal_size = 100
            cal_x1 = width // 2 - cal_size // 2
            cal_y1 = height // 2 - cal_size // 2
            cal_x2 = cal_x1 + cal_size
            cal_y2 = cal_y1 + cal_size
            
            cv2.rectangle(frame, (cal_x1, cal_y1), (cal_x2, cal_y2), (0, 255, 255), 2)
            cv2.putText(frame, "Place object here", (cal_x1, cal_y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Press 's' to sample color", (cal_x1, cal_y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'x' to exit calibration", (cal_x1, cal_y2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display the calibration frame
            cv2.imshow("Subway Surfers Control", frame)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Sample the color in the rectangle
                roi = hsv[cal_y1:cal_y2, cal_x1:cal_x2]
                
                # Calculate average color values
                avg_hue = np.mean(roi[:, :, 0])
                avg_sat = np.mean(roi[:, :, 1])
                avg_val = np.mean(roi[:, :, 2])
                
                # Set thresholds with tolerance
                hue_tolerance = 15
                lower_color = np.array([max(0, avg_hue - hue_tolerance), 50, 50], dtype=np.uint8)
                upper_color = np.array([min(179, avg_hue + hue_tolerance), 255, 255], dtype=np.uint8)
                
                print(f"Color sampled: H={avg_hue:.1f}, S={avg_sat:.1f}, V={avg_val:.1f}")
                print(f"Color range set: {lower_color} to {upper_color}")
                print("Calibration complete! Now tracking your colored object.")
                calibration_mode = False
                
            elif key == ord('x'):
                print("Exiting calibration mode using default values.")
                calibration_mode = False
                
            elif key == ord('q'):
                break
                
        else:  # Normal tracking mode
            # Create mask for color detection
            mask = cv2.inRange(hsv, lower_color, upper_color)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (9, 9), 0)
            
            # Find the largest contour
            largest_contour = find_largest_contour(mask)
            
            # Get the center of the contour
            center = get_contour_center(largest_contour)
            
            if center:
                # Draw a circle at the center of the contour
                cv2.circle(frame, center, 10, (0, 255, 255), -1)
                
                # Determine action based on position
                action = determine_action(center, thresholds)
                
                # Display current action
                cv2.putText(frame, f"Action: {action}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Only execute action if cooldown has elapsed and not in NEUTRAL or NONE state
                current_time = time.time()
                if (current_time - last_action_time > action_cooldown and 
                    action not in ["NEUTRAL", "NONE"]):
                    executed = execute_game_action(action)
                    if executed:
                        last_action_time = current_time
                        cv2.putText(frame, f"Executed: {executed}", (width - 200, height - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the mask in a separate window for debugging
            cv2.imshow("Color Mask", mask)
            
            # Show the main frame
            cv2.imshow("Subway Surfers Control", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("Entering calibration mode...")
                calibration_mode = True
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated")

if __name__ == "__main__":
    main()