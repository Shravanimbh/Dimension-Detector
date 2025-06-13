import cv2    #open source computer vision library
import numpy as np
import argparse   #to write user frndly command line interface
import datetime   

def calculate_dimensions(image_path, ref_object_width_cm):
    """
    Calculates the dimensions of a target object in an image using a reference object of known width.
    """
    # 1. READ THE IMAGE

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at '{image_path}'")
        return

    # 2. PREPROCESSING
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #simplifies RGB in one single channel
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) #makes the image smooth therefore boundaries stand out clearly

    # Canny - edge detection algo
    edged = cv2.Canny(blurred, 50, 100)
    
    # Use morphology cuz there might be gaps in outline to fill those
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


    # 3. FIND AND SORT CONTOURS

    # to find outermost contours (outline) not the inner ones
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 2:
        print("Error: Not enough contours found. Try adjusting preprocessing parameters.")
        return

    # Sort contours by area from largest to smallest. - largest will be the main object 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    ref_contour = None
    ref_box_pts = None
    pixels_per_cm = None

    # 4. IDENTIFY THE REFERENCE OBJECT -> credit card can be detected by aspect ratio (default aspect ratio approx 1.586) loop over to find approx 1.5 to 1.7

    for cnt in contours:
        box = cv2.minAreaRect(cnt)
        (w, h) = box[1]

        #dividing the larger side by the smaller side -> aspect ratio
        try:
            pixel_width = min(w, h)
            pixel_height = max(w, h)
            aspect_ratio = pixel_height / pixel_width
        except ZeroDivisionError:
            continue

        if 1.5 < aspect_ratio < 1.7:
            ref_contour = cnt
            ref_box_pts = cv2.boxPoints(box) 
            pixels_per_cm = pixel_width / ref_object_width_cm
            print(f"[INFO] Reference object found. Pixels/cm: {pixels_per_cm:.2f}")
            break 

    if pixels_per_cm is None:
        print("Error: Reference object not found. Make sure it is clearly visible and has an aspect ratio between 1.5 and 1.7.")
        return

    # 5. IDENTIFY AND MEASURE THE TARGET OBJECT

    # Since the contours are sorted by area, the largest one is the target.
    target_contour = contours[0]
    
    # Safety check: if the largest object was the reference card, the target is the second largest.
    if cv2.contourArea(target_contour) < cv2.contourArea(ref_contour) * 1.1:
      target_contour = contours[1]

    # 6. CALCULATE AND DISPLAY THE TARGET'S DIMENSIONS
    
    target_box = cv2.minAreaRect(target_contour)
    target_box_pts = cv2.boxPoints(target_box)
    (target_w_px, target_h_px) = target_box[1]
    #converted into pixels_per_cm by dividing it by pixels_per_cm of reference object
    target_width_cm = target_w_px / pixels_per_cm
    target_height_cm = target_h_px / pixels_per_cm
    dim1, dim2 = sorted([target_width_cm, target_height_cm])
    dim_text = f"{dim1:.2f}cm x {dim2:.2f}cm"
    print(f"[RESULT] Target Object Dimensions: {dim_text}")

    # 7. DRAW RESULTS ON THE IMAGE

    # to draw the reference box in GREEN and the target box in BLUE
    cv2.drawContours(image, [ref_box_pts.astype(int)], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [target_box_pts.astype(int)], -1, (255, 0, 0), 2)
    text_x = int(target_box_pts[:, 0].min())
    text_y = int(target_box_pts[:, 1].min()) - 15
    cv2.putText(image, dim_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 8. SAVE AND PREPARE FOR DISPLAY

    output_path = f"output_measured.png"
    cv2.imwrite(output_path, image) # Saves the full-resolution image
    print(f"[INFO] Saved measured image to {output_path}")

    # Log the results to a file

    with open("dimensions_log.txt", "a") as log_file:
        log_file.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} - {image_path} - {dim_text}\n")

    
    # === RESIZE IMAGE FOR DISPLAY TO FIT ON SCREEN ===
   
    max_display_height = 900
    h, w = image.shape[:2]

    if h > max_display_height:
        ratio = max_display_height / float(h)
        new_width = int(w * ratio)
        display_image = cv2.resize(image, (new_width, max_display_height))
    else:
        display_image = image

    cv2.imshow("Measured Dimensions (Resized to Fit Screen)", display_image)
    # =================================================================

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure object dimensions in an image using a reference object.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-w", "--ref_width", type=float, required=True, help="Width of the reference object's SHORTER side in cm (e.g., 5.4 for a credit card).")
    args = parser.parse_args()

    calculate_dimensions(args.image, args.ref_width)