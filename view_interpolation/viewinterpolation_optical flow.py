import cv2
import numpy as np

# Step 1: Load two consecutive color frames (left and right views)
img1 = cv2.imread(r"C:\Users\DELL\Desktop\github_cv\view_interpolation\img1.jpeg")   # First view
img2 = cv2.imread(r"C:\Users\DELL\Desktop\github_cv\view_interpolation\img2.jpeg")  # Second view

# Check if images are loaded
if img1 is None or img2 is None:
    print("❌ Could not load one or both images!")
    exit()

# Convert to grayscale for optical flow
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Step 2: Compute Dense Optical Flow (Farneback)
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2,
    flags=0
)

# Step 3: Extrapolate a new view (simulate a new camera position)
def extrapolate_view(img, flow, alpha=1.0):
    h, w = flow.shape[:2]
    # Generate grid of pixel indices
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Calculate new pixel locations using flow and alpha
    map_x = (x + alpha * flow[..., 0]).astype(np.float32)
    map_y = (y + alpha * flow[..., 1]).astype(np.float32)
    # Warp image using the remapped coordinates
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

# Step 4: Generate multiple extrapolated views
alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Simulates motion from left to right
views = []

for a in alphas:
    extrapolated = extrapolate_view(img1, flow, alpha=a)
    views.append(extrapolated)

# Step 5: Display results
for i, view in enumerate(views):
    cv2.imshow(f"Extrapolated View {alphas[i]:+}", view)

cv2.waitKey(0)
cv2.destroyAllWindows()
