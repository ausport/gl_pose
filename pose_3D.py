import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet import shapes, text

import numpy as np
import cv2
import os
import json
import math


RENDER_FPS = 62
RENDER_INTERVAL = 1 / RENDER_FPS
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1280


BALL_PATH = "./samples/kick_to_mark_2_A2_pose_lpf_5Hz_ball.json"
FILE_PATH = "./samples/kick_to_mark_2_A2_pose_lpf_5Hz.json"

BALL_PATH = "./samples/short_kick_3_A2_pose_lpf_5Hz_ball.json"
FILE_PATH = "./samples/short_kick_3_A2_pose_lpf_5Hz.json"


file_root, _ = os.path.splitext(FILE_PATH)
# VIDEO_WRITER = cv2.VideoWriter(file_root + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), RENDER_FPS, (WINDOW_WIDTH, WINDOW_HEIGHT))

# Play the animation automatically
AUTO_PLAY = False

# Automatically cycle between preset views
CYCLE_VIEWS = False
CURRENT_VIEW = 0

# Show the keystroke information
SHOW_INFO = True

# Record the sample to a video
RECORD_MODE = False

# Trail effect 
TOGGLE_TRAILS = True
N_TRAIL_LENGTH = 15


ASPSET_KEYPOINT_NAMES = np.array([
    'right_ankle', 'right_knee', 'right_hip', 'right_wrist', 'right_elbow', 'right_shoulder',
    'left_ankle', 'left_knee', 'left_hip', 'left_wrist', 'left_elbow', 'left_shoulder',
    'head_top', 'head', 'neck', 'spine', 'pelvis', 'right_toe_base', 'right_heel', 'left_toe_base', 'left_heel'
])

ASPSET_JOINT_PAIRS = np.array([
    ['right_ankle', 'right_knee'],
    ['right_knee', 'right_hip'],
    ['right_hip', 'pelvis'],
    ['left_hip', 'pelvis'],
    ['left_knee', 'left_hip'],
    ['left_ankle', 'left_knee'],
    ['pelvis', 'spine'],
    ['spine', 'neck'],
    ['neck', 'head'],
    ['head', 'head_top'],
    ['right_wrist', 'right_elbow'],
    ['right_elbow', 'right_shoulder'],
    ['right_shoulder', 'neck'],
    ['left_shoulder', 'neck'],
    ['left_wrist', 'left_elbow'],
    ['left_elbow', 'left_shoulder'],
    ['right_ankle', 'right_heel'],
    ['right_heel', 'right_toe_base'],
    ['left_ankle', 'left_heel'],
    ['left_heel', 'left_toe_base'],
])

SHOW_TRAILS = ASPSET_KEYPOINT_NAMES

pose_data = None
joint_wise_axis_means = np.array([0, 0, 0])
global_axis_means = np.array([0, 0, 0])
global_axis_max = np.array([0, 0, 0])
global_axis_min = np.array([0, 0, 0])

frame = 0
ball_rotation = 40

# Get the ball data
# TODO - create a ball tracking json file that follows the same format as the pose data
with open(BALL_PATH, 'r') as f:
    json_data = f.read()

ball_positions = []
ball_data = json.loads(json_data)
# Iterate through each pose in the data looking for the extra ball
for pose in ball_data:
    keypoints = pose['data']['data']  # Assuming this is the list of 2D keypoints
    if len(keypoints) > 0:
        # Add a third axis (z = 0) for each keypoint in each instance
        keypoints_with_z = [[[x, y, 0] for x, y in instance] for instance in keypoints]

        for instance in keypoints_with_z:
            if len(instance) == 22:  # Check if the instance contains a ball keypoint
                ball_positions.append(instance.pop(21))  # Extract and remove the ball keypoint

# Convert lists to NumPy arrays for easier manipulation
ball_positions_np = np.array(ball_positions)  # Contains only ball keypoints

with open(FILE_PATH, 'r') as f:
    json_data = f.read()

pose_data = json.loads(json_data)

# Initialize lists to accumulate data
all_keypoints = []

# Iterate through each pose in data
for pose in pose_data:
    keypoints = pose['data']['data']  # Assuming this is the list of 2D keypoints
    if len(keypoints) > 0:
        # Add a third axis (z = 0) for each keypoint in each instance
        keypoints_with_z = [[[x, y, 0] for x, y in instance] for instance in keypoints]
        all_keypoints.extend(keypoints_with_z)

# Convert all_keypoints to a NumPy array for easier manipulation
data_np = np.array(all_keypoints)

# Calculate mean along axis 0 to get global mean of each (x, y, z) tuple
global_axis_means = np.mean(data_np, axis=(0, 1))  # Global mean of x, y, z

# Center the data by subtracting global_axis_means
centered_data = data_np - global_axis_means  # Broadcasting subtracts [x_mean, y_mean, z_mean] from every point

# Update global metrics for the centered data
global_axis_max = np.max(centered_data, axis=(0, 1))
global_axis_min = np.min(centered_data, axis=(0, 1))

# print("Global_axis_means (x, y, z):", global_axis_means)
# print("Global_axis_min (x, y, z):", global_axis_min)
# print("Global_axis_max (x, y, z):", global_axis_max)


glEnable(GL_DEPTH_TEST)
glEnable(GL_LINE_SMOOTH)

viewpoints = [{"zoom_factor": 15.5, "rotation_angle_horizontal": 5.3, "rotation_angle_vertical": -0.4299, "camera_position": [20, 10, 5], "look_at": [235, -5, 0]},
              {"zoom_factor": 27.27, "rotation_angle_horizontal": 5.3, "rotation_angle_vertical": -0.4299, "camera_position": [20, 10, 5], "look_at": [235, -5, 0]},
              {"zoom_factor": 14.289, "rotation_angle_horizontal": 0.71, "rotation_angle_vertical": -0.6999, "camera_position": [20, 18, 5], "look_at": [245, -15, 0]},
              {"zoom_factor": 22.869, "rotation_angle_horizontal": 0.71, "rotation_angle_vertical": -0.6999, "camera_position": [20, 18, 5], "look_at": [245, -15, 0]},
             ]

viewpoint = viewpoints[CURRENT_VIEW]

zoom_factor, rotation_angle_horizontal, rotation_angle_vertical, camera_position, look_at = viewpoint.values()

SURFACE_LENGTH = 200
SURFACE_WIDTH = 200

MAX_VERTICAL_ANGLE = math.pi / 2 - 0.1  # Just below straight up
MIN_VERTICAL_ANGLE = -MAX_VERTICAL_ANGLE  # Just below straight down

window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, resizable=False)

sample_label = pyglet.text.Label(
    os.path.basename(FILE_PATH),
    font_name='Arial',
    font_size=20,
    x= window.width // 2,
    y= window.height - 30,
    anchor_x='center',
    anchor_y='center',
    color=(255, 255, 255, 255)
)

instructions_label = pyglet.text.Label(
    "Play/Pause: SPACE\nFrame Forward: RIGHT ARROW\nFrame Backward: LEFT ARROW\n"
    "Step Forward: SHIFT+RIGHT ARROW\nStep Backward: SHIFT+LEFT ARROW\n"
    "Record Video: R\nToggle Joint Trails: T\nViewpoints: [0-3]\nClose: ESC",
    font_size=12,
    x=10,
    y=10,
    multiline=True,
    width=400,
    anchor_x="left",
    anchor_y="bottom",
)

def time_label_with_value(t):
    return text.Label(
        f"{t:.2f}",
        font_name='Arial',
        font_size=14,
        x=window.width - 25,
        y=10,
        multiline=False,
        width=100,
        anchor_x="right",
        anchor_y="bottom",
    )
    
def draw_surface():
    # Define the size of each square
    square_length = SURFACE_LENGTH / 8
    square_width = SURFACE_WIDTH / 8

    glColor3f(0.6, 0.6, 0.6)  # Light grey color for the grid
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glBegin(GL_LINES)
    
    # Grid spacing based on 1/8th of the court width
    grid_spacing = SURFACE_WIDTH / 10
    
    max_distance = SURFACE_LENGTH * 2
    fade_distance = SURFACE_LENGTH 
    
    # Draw vertical lines
    for x in range(-int(max_distance / grid_spacing), int(max_distance / grid_spacing) + 1):
        distance = abs(x * grid_spacing)
        alpha = max(0.0, 1.0 - (distance - fade_distance) / fade_distance)
        glColor4f(0.25, 0.25, 0.25, alpha)
        glVertex3f(x * grid_spacing, -max_distance / 1, 0)
        glVertex3f(x * grid_spacing, max_distance / 1, 0)
    
    # Draw horizontal lines
    for y in range(-int(max_distance / grid_spacing), int(max_distance / grid_spacing) + 1):
        distance = abs(y * grid_spacing)
        alpha = max(0.0, 1.0 - (distance - fade_distance) / fade_distance)
        glColor4f(0.25, 0.25, 0.25, alpha)
        glVertex3f(-max_distance / 1, y * grid_spacing, 0)
        glVertex3f(max_distance / 1, y * grid_spacing, 0)
    
    glEnd()
    glDisable(GL_BLEND)


class PoseRenderer:
    def __init__(self, pose, trail_data=[], player_id=0):
        self.pose = pose
        self.trail_data = trail_data
        self.origin = [0, 0]
        self.player_id = player_id

    @staticmethod
    def ensure_3d(coords):
        """Ensure that the input coordinates have a third dimension (z)."""
        if len(coords) == 2:
            return [coords[0], coords[1], 0]  # Return as a list
        return list(coords)  # Ensure consistency in the output


    def draw_trail_line(self, trail_points, width=1.0, r=1.0, g=1.0, b=1.0):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        num_points = len(trail_points)
        if num_points < 2:
            return
        
        glBegin(GL_TRIANGLE_STRIP)
        for i, point in enumerate(trail_points):
            x, y, z = self.ensure_3d(point)

            x = x + global_axis_means[0]
            y = (y - global_axis_means[1]) + global_axis_min[1]
            z = z - global_axis_means[2]
            
            a = i / (num_points - 1)  # Calculate opacity based on index
            glColor4f(r, g, b, a)
            
            if i == 0:
                dx, dy, dz = trail_points[i + 1][0] - x, trail_points[i + 1][1] - y, trail_points[i + 1][2] - z
            elif i == num_points - 1:
                dx, dy, dz = x - trail_points[i - 1][0], y - trail_points[i - 1][1], z - trail_points[i - 1][2]
            else:
                dx, dy, dz = trail_points[i + 1][0] - trail_points[i - 1][0], trail_points[i + 1][1] - trail_points[i - 1][1], trail_points[i + 1][2] - trail_points[i - 1][2]
            
            length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            if length > 0:
                dx /= length
                dy /= length
                dz /= length

            nx, ny, nz = -dy, dx, 0
            x1, y1, z1 = x + width * nx, y + width * ny, z + width * nz
            x2, y2, z2 = x - width * nx, y - width * ny, z - width * nz
            
            glVertex3f(x1/20, -y1/20, z1/20)
            glVertex3f(x2/20, -y2/20, z2/20)
        glEnd()
    
    def draw_joint_center(self, x, y, z, radius=0.3, slices=16, stacks=16):
        x = x + global_axis_means[0]
        y = (y - global_axis_means[1]) + global_axis_min[1]
        z = z - global_axis_means[2]
        
        glPushMatrix()
        glTranslatef(x/20, -y/20, z/20)  # Flip y-axis here by negating y

        # Set emissive material properties (glow color)
        emissive_color = (1.0 , 1.0, 0.0, 0.5)  # Example: orange glow
        glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat * 4)(*emissive_color))

        # Draw the sphere
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

        # Reset emissive material to avoid affecting other objects
        no_emission = (0.0, 0.0, 0.0, 1.0)
        glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat * 4)(*no_emission))
        glPopMatrix()

    def draw_limb_length(self, from_tuple, to_tuple, line_width=2.0):
        from_x, from_y, from_z = self.ensure_3d(from_tuple)
        to_x, to_y, to_z = self.ensure_3d(to_tuple)

        from_x = from_x + global_axis_means[0]
        from_y = (from_y - global_axis_means[1]) + global_axis_min[1]
        from_z = from_z - global_axis_means[2]
        
        to_x = to_x + global_axis_means[0]
        to_y = (to_y - global_axis_means[1]) + global_axis_min[1]
        to_z = to_z - global_axis_means[2]

        glColor3f(0.4, 0.4, 0.0)  # Light grey color for the line
        glLineWidth(line_width)
        glBegin(GL_LINES)
        glVertex3f(from_x / 20, -from_y / 20, from_z / 20)
        glVertex3f(to_x / 20, -to_y / 20, to_z / 20)
        glEnd()
        glLineWidth(1.0)  # Reset line width to default

    def draw(self):
        if len(self.pose) > 0:
            # Draw the joint centre trails first
            if TOGGLE_TRAILS:
                for points in self.trail_data:
                    if len(self.trail_data[points]) > 0:
                        sorted_data = sorted(self.trail_data[points], key=lambda x: x['frame'])
                        sorted_pos = [self.ensure_3d(item['pos']) for item in sorted_data]
                        self.draw_trail_line(sorted_pos)
            
            keypoint_index = {name: idx for idx, name in enumerate(ASPSET_KEYPOINT_NAMES)}
            
            for pair in ASPSET_JOINT_PAIRS:
                from_keypoint, to_keypoint = pair
                from_index = keypoint_index[from_keypoint]
                to_index = keypoint_index[to_keypoint]               
                from_tuple = self.ensure_3d(self.pose[self.player_id][from_index])
                to_tuple = self.ensure_3d(self.pose[self.player_id][to_index])
                # Draw segment
                self.draw_limb_length(from_tuple, to_tuple)

            enable_lighting()
            for pos_group in self.pose:
                for pos in pos_group:
                    pos = self.ensure_3d(pos)
                    self.draw_joint_center(pos[0], pos[1], pos[2])
            disable_lighting()

class BallRenderer:
    def __init__(self, ball_positions, ball_trail_data=[]):
        self.ball_positions = ball_positions
        self.ball_trail_data = ball_trail_data
        self.origin = [0, 0]

    @staticmethod
    def ensure_3d(coords):
        """Ensure that the input coordinates have a third dimension (z)."""
        if len(coords) == 2:
            return [coords[0], coords[1], 0]  # Return as a list
        return list(coords)  # Ensure consistency in the output


    def draw_trail_line(self, trail_points, width=1.0, r=1.0, g=1.0, b=1.0):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        num_points = len(trail_points)
        if num_points < 2:
            return

        glBegin(GL_TRIANGLE_STRIP)
        for i, point in enumerate(trail_points):
            x, y, z = self.ensure_3d(point)

            x = x + global_axis_means[0]
            y = (y - global_axis_means[1]) + global_axis_min[1]
            z = z - global_axis_means[2]

            a = i / (num_points - 1)  # Calculate opacity based on index
            glColor4f(r, g, b, a)

            if i == 0:
                dx, dy, dz = trail_points[i + 1][0] - x, trail_points[i + 1][1] - y, trail_points[i + 1][2] - z
            elif i == num_points - 1:
                dx, dy, dz = x - trail_points[i - 1][0], y - trail_points[i - 1][1], z - trail_points[i - 1][2]
            else:
                dx, dy, dz = trail_points[i + 1][0] - trail_points[i - 1][0], trail_points[i + 1][1] - trail_points[i - 1][1], trail_points[i + 1][2] - trail_points[i - 1][2]

            length = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            if length > 0:
                dx /= length
                dy /= length
                dz /= length

            nx, ny, nz = -dy, dx, 0
            x1, y1, z1 = x + width * nx, y + width * ny, z + width * nz
            x2, y2, z2 = x - width * nx, y - width * ny, z - width * nz

            glVertex3f(x1/20, -y1/20, z1/20)
            glVertex3f(x2/20, -y2/20, z2/20)
        glEnd()

    def draw_ball_center(self, x, y, z, radius=0.8, slices=16, stacks=16):
        global ball_rotation
        global frame

        x = x + global_axis_means[0]
        y = (y - global_axis_means[1]) + global_axis_min[1]
        z = z - global_axis_means[2]

        glPushMatrix()
        glTranslatef(x / 20, -y / 20, z / 20)  # Flip y-axis here by negating y

        _r = 0.4 if frame < 50 else ball_rotation
        glRotatef(_r, 1.0, 0.0, 0.0)  # Rotate around the y-axis

        emissive_color = (0.5, 0.1, 0.0, 0.1)  # Example: orange glow
        glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat * 4)(*emissive_color))

        # Scale to create an oval shape (elongate along one axis, e.g., y-axis)
        glScalef(1.0, 1.5, 1.0)  # Adjust these values for desired proportions

        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)  # Sphere becomes an oval due to scaling
        gluDeleteQuadric(quadric)

        # Reset emissive material to avoid affecting other objects
        no_emission = (0.0, 0.0, 0.0, 1.0)
        glMaterialfv(GL_FRONT, GL_EMISSION, (GLfloat * 4)(*no_emission))
        glPopMatrix()

        ball_rotation += 8.0  # Increment the rotation angle

    def draw(self):
        if len(self.ball_positions) > 0:
            # Draw the ball trails first
            if TOGGLE_TRAILS:
                for points in self.ball_trail_data:
                    if len(self.ball_trail_data[points]) > 0:
                        sorted_data = sorted(self.ball_trail_data[points], key=lambda x: x['frame'])
                        sorted_pos = [self.ensure_3d(item['pos']) for item in sorted_data]
                        self.draw_trail_line(sorted_pos)

            enable_lighting()
            pos = self.ball_positions
            print("Ball position:", pos)
            pos = self.ensure_3d(pos)
            self.draw_ball_center(pos[0], pos[1], pos[2])
            disable_lighting()


def enable_lighting():
    glEnable(GL_LIGHTING)

    # Light 0 properties
    glEnable(GL_LIGHT0)
    light_diffuse0 = (1.0, 1.0, 1.0, 1.0)
    light_ambient0 = (0.1, 0.1, 0.1, 1.0)
    light_position0 = (500.0, 500.0, 1000.0, 1.0)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 4)(*light_diffuse0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat * 4)(*light_ambient0))
    glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat * 4)(*light_position0))

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

    global_ambient = (0.1, 0.1, 0.1, 1.0)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (GLfloat * 4)(*global_ambient))


def disable_lighting():
    glDisable(GL_LIGHTING)
    glDisable(GL_LIGHT0)
    glDisable(GL_LIGHT1)
    glDisable(GL_COLOR_MATERIAL)
    

# Pre-baked view points..
keystroke_to_index = {
    pyglet.window.key._0: 0,
    pyglet.window.key._1: 1,
    pyglet.window.key._2: 2,
    pyglet.window.key._3: 3
}

def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global zoom_factor
    zoom_factor += scroll_y * 1.1
    zoom_factor = max(0.1, zoom_factor)  # Limit zoom factor to avoid negative values
    
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global rotation_angle_horizontal, rotation_angle_vertical
    rotation_speed = 0.01
    rotation_angle_horizontal += dx * rotation_speed
    rotation_angle_vertical += dy * rotation_speed
        
def on_key_press(symbol, modifiers):
    global frame
    global zoom_factor, rotation_angle_horizontal, rotation_angle_vertical, camera_position, look_at
    global AUTO_PLAY
    global SHOW_INFO
    global RECORD_MODE
    global TOGGLE_TRAILS
    
    if symbol == pyglet.window.key.LEFT:
        if modifiers & pyglet.window.key.MOD_SHIFT:
            frame = max(0, frame - RENDER_FPS)
        else:
            frame = max(0, frame - 1)

    elif symbol == pyglet.window.key.RIGHT:
        if modifiers & pyglet.window.key.MOD_SHIFT:
            frame = min(len(pose_data), frame + RENDER_FPS)
        else:
            frame = min(len(pose_data), frame + 1)
            
    elif symbol == pyglet.window.key.SPACE:
        AUTO_PLAY = not AUTO_PLAY
        if AUTO_PLAY:
            if frame >= len(pose_data):
                frame = 0
            pyglet.clock.schedule_interval(update, RENDER_INTERVAL)
        else:
            pyglet.clock.unschedule(update)


    elif symbol == pyglet.window.key.A:
        camera_position[0] = camera_position[0] - 1
    elif symbol == pyglet.window.key.D:
        camera_position[0] = camera_position[0] + 1


    elif symbol == pyglet.window.key.Z:
        camera_position[0] = camera_position[0] - 1
    elif symbol == pyglet.window.key.W:
        camera_position[1] = camera_position[1] + 1


    elif symbol == pyglet.window.key.NUM_4:
        look_at[0] = look_at[0] - 10
    elif symbol == pyglet.window.key.NUM_6:
        look_at[0] = look_at[0] + 10

    elif symbol == pyglet.window.key.NUM_8:
        look_at[1] = look_at[1] + 10
    elif symbol == pyglet.window.key.NUM_2:
        look_at[1] = look_at[1] - 10


    elif symbol == pyglet.window.key.Z:
        camera_position[0] = camera_position[0] - 1
    elif symbol == pyglet.window.key.W:
        camera_position[1] = camera_position[1] + 1


    elif symbol == pyglet.window.key.T:
        TOGGLE_TRAILS = not TOGGLE_TRAILS
        
    elif symbol == pyglet.window.key.I:
        SHOW_INFO = not SHOW_INFO
    
    elif symbol == pyglet.window.key.R:
        RECORD_MODE = not RECORD_MODE
        if RECORD_MODE:
            # Zero the clock, enable a file write and hide the instructions.
            SHOW_INFO = False
            AUTO_PLAY = False
            frame = 0
            pyglet.clock.schedule_once(update, RENDER_INTERVAL)
    
    elif symbol == pyglet.window.key.ESCAPE or symbol == pyglet.window.key.Q:
        window.close()

    elif symbol in keystroke_to_index:
        index = keystroke_to_index[symbol]
        viewpoint = viewpoints[index]
        zoom_factor, rotation_angle_horizontal, rotation_angle_vertical, camera_position, look_at = viewpoint.values()

        
def update_camera():
    global zoom_factor
    glLoadIdentity()
    
    # Update the position of the camera
    x = camera_position[0] * zoom_factor * math.cos(rotation_angle_horizontal)
    y = camera_position[1] * zoom_factor * math.sin(rotation_angle_horizontal)
    z = max(camera_position[2] * zoom_factor * -rotation_angle_vertical, 30)    

    gluLookAt(x, y, z,                              # Camera position
              look_at[0], look_at[1], look_at[2],   # Look at the origin
              0, 0, 1)                              # Up vector

    print(f"zoom_factor: {zoom_factor}, rotation_angle_horizontal: {rotation_angle_horizontal}, rotation_angle_vertical: {rotation_angle_vertical}")
    print(f"camera_position x: {camera_position[0]}, camera_position y: {camera_position[1]}")
    print(f"look_at x: {look_at[0]}, look_at y: {look_at[1]}")

@window.event
def on_draw():
    global zoom_factorpos_y
    global AUTO_PLAY
    global CYCLE_VIEWS
    global RECORD_MODE
    global pose_data, ball_positions

    glClearColor(0.2, 0.2, 0.2, 1)  # Dark grey background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, window.width / window.height, 10, 10000)
    glMatrixMode(GL_MODELVIEW)
    
    update_camera()

    # Draw the surface and labels
    draw_surface()

    # Rotate the modelview matrix to swap the x and z axes
    glRotatef(90, 1, 0, 0)  # Rotate 90 degrees around the x-axis

    # Draw the poses and their historical trails
    current_pose = pose_data[frame]
    current_keypoints = current_pose['data']['data']

    if len(current_keypoints) > 0:

        current_ball = ball_positions_np[frame]
        print(current_ball)

        for player_id in [0, 1]:
            # Assemble the trail data from the current frame
            trail_data = {keypoint: [] for keypoint in SHOW_TRAILS}

            for frame_idx in range(max(0, frame-20), frame):
                pose_for_frame = pose_data[frame_idx]
                if len(pose_for_frame['data']['data']) > 0:
                    trail_keypoints = pose_for_frame['data']['data'][player_id]

                    # Update the joint data dictionary
                    for keypoint, position in zip(SHOW_TRAILS, trail_keypoints):
                        if not any(entry["frame"] == frame_idx for entry in trail_data[keypoint]):
                            trail_data[keypoint].append({"frame": frame_idx, "pos": tuple(position)})

                            # Keep only the last n tuples
                            if len(trail_data[keypoint]) > N_TRAIL_LENGTH:
                                trail_data[keypoint].pop(0)

            # Draw the pose
            if len(current_keypoints) >= 0:
                pose_model = PoseRenderer(current_keypoints, trail_data, player_id)
                pose_model.draw()
                ball_trail_data = []
                ball_model = BallRenderer(ball_positions_np[frame], ball_trail_data)
                ball_model.draw()



    # Switch to 2D mode for on-screen info
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window.width, 0, window.height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    sample_label.draw()
    instructions_label.draw()

    time_label = time_label_with_value(frame/RENDER_FPS)
    time_label.draw()

    if SHOW_INFO:
        print("Showing instructions")
        instructions_label.draw()
                
    if RECORD_MODE:
        # Convert the window buffer to a numpy array
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        data = image_data.get_data()

        # Convert to numpy array
        image_as_np = np.frombuffer(data, dtype=np.uint8).reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 4)

        # Convert RGBA to BGR (OpenCV uses BGR)
        frame_bgr = cv2.flip(cv2.cvtColor(image_as_np, cv2.COLOR_RGBA2BGR), 0)

        # Write frame to video
        VIDEO_WRITER.write(frame_bgr)

        pyglet.clock.schedule_once(update, RENDER_INTERVAL)

        if frame >= len(ball_positions) or RECORD_MODE == False:
            # Wrap it up..
            VIDEO_WRITER.release()
            RECORD_MODE = False
        
def update(dt):
    global frame
    global CURRENT_VIEW, RECORD_MODE
    global zoom_factor, rotation_angle_horizontal, rotation_angle_vertical
    
    if frame <= len(pose_data)-2:
        frame += 1
    else:
        frame = 0
        if CYCLE_VIEWS:
            if CURRENT_VIEW == 3:
                RECORD_MODE = False
                pyglet.clock.unschedule(update)
            
            CURRENT_VIEW = CURRENT_VIEW + 1 if CURRENT_VIEW < 3 else 0
            viewpoint = viewpoints[CURRENT_VIEW]
            zoom_factor, rotation_angle_horizontal, rotation_angle_vertical, camera_position, look_at = viewpoint.values()


# Event handlers..
window.on_mouse_scroll = on_mouse_scroll
window.on_mouse_drag = on_mouse_drag
window.on_draw = on_draw
window.on_key_press = on_key_press

AUTO_PLAY = False

if AUTO_PLAY:
    pyglet.clock.schedule_interval(update, RENDER_INTERVAL)
    
pyglet.app.run()

VIDEO_WRITER.release()

