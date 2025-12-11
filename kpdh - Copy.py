"""
demon_game.py

Standalone webcam game using OpenCV.

Controls:
 - Move a "cake pop" (any hand/fast motion) to touch corner demons (+1 each).
 - Avoid the falling demon; if it hits your face -> GAME OVER.
 - If the falling demon leaves the bottom without hitting you -> +5 points.
 - Press 'q' to quit.
"""

import cv2
import time
import random

# ---------- Config ----------
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Demon image sizes (in pixels) ---
CORNER_DEMON_W = 60      # width of corner demon image
CORNER_DEMON_H = 60      # height of corner demon image

# How often demons appear & disappear
CORNER_SPAWN_INTERVAL = 3.0
CORNER_DEMON_LIFETIME = 6.0

# Falling demon image size
FALLING_DEMON_W = 80
FALLING_DEMON_H = 80
FALLING_SPAWN_INTERVAL = 7.0
FALLING_SPEED = 180

# Motion detection thresholds (unchanged)
MOTION_THRESHOLD = 25
MIN_MOTION_AREA = 250

FONT = cv2.FONT_HERSHEY_SIMPLEX
# ---------- Helper functions ----------
def rects_intersect(r1, r2):
    # r = (x, y, w, h)
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

# Overlay PNG with alpha channel onto the frame
def overlay_image(background, overlay, x, y):
    """
    Draw `overlay` (BGRA or BGR) onto `background` (BGR)
    at position (x, y). Handles transparency.
    """
    h, w = overlay.shape[:2]

    if x + w > background.shape[1] or y + h > background.shape[0]:
        return  # image would go out of bounds

    if overlay.shape[2] == 4:  # PNG with alpha
        b, g, r, a = cv2.split(overlay)
        alpha = a.astype(float) / 255.0
    else:
        b, g, r = cv2.split(overlay)
        alpha = np.ones((h, w), dtype=float)

    for c, channel in enumerate([b, g, r]):
        background[y:y+h, x:x+w, c] = (
            background[y:y+h, x:x+w, c] * (1 - alpha) +
            channel * alpha
        )

# ---------- Initialize ----------
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade.")

prev_gray = None
score = 0
game_over = False

# ---------- Load Demon Images ----------
corner_demon_img = cv2.imread("corner_demon.jpg", cv2.IMREAD_UNCHANGED)
falling_demon_img = cv2.imread("falling_demon.png", cv2.IMREAD_UNCHANGED)

if corner_demon_img is None:
    raise RuntimeError("Could not load corner_demon.jpg")

if falling_demon_img is None:
    raise RuntimeError("Could not load falling_demon.png")

# Resize to match your config values
corner_demon_img = cv2.resize(corner_demon_img, (CORNER_DEMON_W, CORNER_DEMON_H))
falling_demon_img = cv2.resize(falling_demon_img, (FALLING_DEMON_W, FALLING_DEMON_H))

# Corner demons stored as dicts: {'corner': 'tl'/'tr'/'bl'/'br', 'rect':(x,y,w,h), 'spawned':t, 'alive':True}
corner_demons = []
last_corner_spawn = 0.0

# Falling demon: dict or None with keys: {'x','y','w','h','spawn_time','active'}
falling = None
last_falling_spawn = 0.0

# For smooth motion timing
prev_time = time.time()

# Corners helper
def corner_rect(corner, w, h):
    if corner == 'tl':
        x, y = 10, 10
    elif corner == 'tr':
        x, y = FRAME_WIDTH - w - 10, 10
    elif corner == 'bl':
        x, y = 10, FRAME_HEIGHT - h - 10
    elif corner == 'br':
        x, y = FRAME_WIDTH - w - 10, FRAME_HEIGHT - h - 10
    return (x, y, w, h)

# Spawn one corner demon in a random available corner
def spawn_corner_demon(now):
    available = [c for c in corner_list if all(d['corner'] != c or not d['alive'] for d in corner_demons)]
    if not available:
        return
    c = random.choice(available)
    r = corner_rect(c, CORNER_DEMON_W, CORNER_DEMON_H)
    corner_demons.append({'corner': c, 'rect': r, 'spawned': now, 'alive': True})

def spawn_falling(now):
    x = (FRAME_WIDTH - FALLING_DEMON_W) // 2
    y = -FALLING_DEMON_H * 3
    return {'x': x, 'y': y, 'w': FALLING_DEMON_W, 'h': FALLING_DEMON_H, 'spawn_time': now, 'active': True}

# ---------- Main loop ----------
print("Starting Demon Game. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize to fixed size (optional)
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    now = time.time()
    dt = now - prev_time
    prev_time = now

    # Gray + blur for motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

    if prev_gray is None:
        prev_gray = gray_blur.copy()

    # Motion mask
    diff = cv2.absdiff(prev_gray, gray_blur)
    _, motion_mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    # Find contours of motion
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_regions = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_MOTION_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        motion_regions.append((x, y, w, h))
        # draw motion (for debugging, faint)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    # Face detection (detectMultiScale expects equalized or grayscale)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    face_rect = None
    if len(faces) > 0:
        # use the biggest detected face (closest to camera)
        faces_sorted = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        (fx, fy, fw, fh) = faces_sorted[0]
        face_rect = (fx, fy, fw, fh)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
        cv2.putText(frame, "Face", (fx, fy - 8), FONT, 0.6, (0, 200, 0), 2)

    # Spawn corner demons periodically
    if now - last_corner_spawn > CORNER_SPAWN_INTERVAL:
        spawn_corner_demon(now)
        last_corner_spawn = now

    # Update corner demons: draw, check motion contact and lifetime
    for d in corner_demons:
        if not d['alive']:
            continue
        x, y, w, h = d['rect']
        # draw demon (filled circle inside rectangle)
        cx, cy = x + w // 2, y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "DEMON", (x, y + h + 15), FONT, 0.5, (0, 0, 255), 1)

        # check motion overlap
        touched = False
        for m in motion_regions:
            if rects_intersect(d['rect'], m):
                touched = True
                break
        if touched:
            d['alive'] = False
            score += 1
            # small particle effect: draw a quick filled circle
            cv2.circle(frame, (cx, cy), int(w * 0.8), (0, 255, 0), thickness=-1)
        else:
            # check lifetime expiry
            if now - d['spawned'] > CORNER_DEMON_LIFETIME:
                d['alive'] = False

    # Spawn falling demon periodically if none active
    if (falling is None or not falling['active']) and (now - last_falling_spawn > FALLING_SPAWN_INTERVAL):
        falling = spawn_falling(now)
        last_falling_spawn = now

    # Update falling demon: move and check collision with face or bottom
    if falling and falling['active']:
        falling['y'] += FALLING_SPEED * dt
        fx = int(falling['x'])
        fy = int(falling['y'])
        fw = int(falling['w'])
        fh = int(falling['h'])
        # draw falling demon as red rectangle
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 180), -1)
        cv2.putText(frame, "DEMON", (fx + 5, fy + 20), FONT, 0.6, (255, 255, 255), 2)

        # check face collision -> GAME OVER
        if fy > 0 and face_rect is not None and rects_intersect(face_rect, (fx, fy, fw, fh)):
            game_over = True
            falling['active'] = False

        # check bottom exit -> +5 points
        if fy > FRAME_HEIGHT:
            falling['active'] = False
            score += 5

    # Draw score
    cv2.putText(frame, f"Score: {score}", (10, 30), FONT, 1.0, (255, 255, 0), 2)

    # If game over, show full-screen message and stop spawning / interactions
    if game_over:
        overlay = frame.copy()
        cv2.putText(overlay, "GAME OVER", (FRAME_WIDTH // 2 - 160, FRAME_HEIGHT // 2 - 20), FONT, 2.0, (0, 0, 255), 6)
        cv2.putText(overlay, f"Final Score: {score}", (FRAME_WIDTH // 2 - 150, FRAME_HEIGHT // 2 + 40), FONT, 1.2, (255, 255, 255), 3)
        # dim background
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show instructions
    cv2.putText(frame, "Press 'q' to quit", (FRAME_WIDTH - 220, FRAME_HEIGHT - 10), FONT, 0.5, (200, 200, 200), 1)

    # Show frame
    cv2.imshow("Demon Game (mirrored)", frame)

    # update prev_gray for next iteration
    prev_gray = gray_blur.copy()

    # key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # if game over, wait for 'q' to quit (but still show frame)
    if game_over:
        # keep showing until user presses q
        # but we should still allow pressing q via waitKey which is already checked above
        pass

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Exited. Final score: {score}")