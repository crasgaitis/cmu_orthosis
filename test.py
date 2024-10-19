import cv2

image = cv2.imread("flask_app/static/images/scale_test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize=3)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)

cv2.imshow("Detected Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
