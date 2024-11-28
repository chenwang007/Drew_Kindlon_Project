import sys
import math
import cv2
import os
import pandas as pd
import sqlite3
import mediapipe as mp
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel,
    QFileDialog, QCheckBox, QDialog, QScrollArea, QGridLayout, QAction, QProgressBar, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QMessageBox, QInputDialog, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from qt_material import apply_stylesheet
import time 

# Database initialization
DB_PATH = "workouts.db"

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            keypoints TEXT
        )
    ''')
    conn.commit()
    conn.close()


class PoseEstimationThread(QThread):
    frame_processed = pyqtSignal(QImage)
    angles_calculated = pyqtSignal(dict)
    speeds_calculated = pyqtSignal(dict)
    processing_complete = pyqtSignal(str)
    progress_updated = pyqtSignal(int)

    def __init__(self, video_path, output_dir, selected_keypoints, interpolation_limit=5):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.selected_keypoints = selected_keypoints
        self.interpolation_limit = interpolation_limit
        self.stop_thread = False
        self.last_speed_update_time = None  # Initialize the variable here

    def run(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

        all_keypoints = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
            "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
            "left_heel", "right_heel", "left_foot_index", "right_foot_index"
        ]

        # Define points for speed calculation
        speed_points = [
            "left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(self.output_dir, f"pose_keypoints_{timestamp}.csv")
        output_video_path = os.path.join(self.output_dir, f"pose_estimation_{timestamp}.avi")

        selected_indices = list(set(
                [all_keypoints.index(kp) for kp in self.selected_keypoints] +
                [all_keypoints.index(kp) for kp in speed_points if kp in all_keypoints]
            ))


        cap = cv2.VideoCapture(self.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        data = []
        frame_number = 0

        # Variables for speed calculation
        previous_coords = {}
        frame_interval = 1  # seconds
        frame_time = 1 / frame_rate
        interval = 1
        self.last_speed_update_time = time.time()  # Initialize before entering the loop

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        while cap.isOpened() and not self.stop_thread:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_data = {'frame': frame_number}

            keypoint_coords = {}
            speeds = {}
            if results.pose_landmarks:
                for idx in selected_indices:
                    landmark = results.pose_landmarks.landmark[idx]
                    x, y = landmark.x * frame_width, landmark.y * frame_height
                    z, visibility = landmark.z, landmark.visibility
                    frame_data[f"{all_keypoints[idx]}_x"] = x
                    frame_data[f"{all_keypoints[idx]}_y"] = y
                    frame_data[f"{all_keypoints[idx]}_z"] = z
                    frame_data[f"{all_keypoints[idx]}_visibility"] = visibility
                    keypoint_coords[all_keypoints[idx]] = (x, y)

                # Speed calculation every `interval` seconds
                current_time = time.time()
                if current_time - self.last_speed_update_time >= interval:
                    speeds = {}
                    for point in speed_points:
                        if point in previous_coords and point in keypoint_coords:
                            prev_x, prev_y = previous_coords[point]
                            curr_x, curr_y = keypoint_coords[point]
                            speed = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) / interval
                            speeds[point] = round(speed, 2)

                    self.last_speed_update_time = current_time
                    self.speeds_calculated.emit(speeds)  # Emit speeds only once per second

                # Update previous coordinates
                for point in speed_points:
                    if point in keypoint_coords:
                        previous_coords[point] = keypoint_coords[point]

                angles = self.calculate_angles(keypoint_coords)
                frame_data.update(angles)
                self.draw_pose_with_angles(frame, keypoint_coords, angles)
                self.angles_calculated.emit(angles)
                self.speeds_calculated.emit(speeds)
            else:
                for idx in selected_indices:
                    frame_data[f"{all_keypoints[idx]}_x"] = None
                    frame_data[f"{all_keypoints[idx]}_y"] = None
                    frame_data[f"{all_keypoints[idx]}_z"] = None
                    frame_data[f"{all_keypoints[idx]}_visibility"] = None

            data.append(frame_data)
            frame_number += 1
            self.progress_updated.emit(int((frame_number / total_frames) * 100))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_qt = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
            self.frame_processed.emit(frame_qt)
            out_video.write(frame)

            QApplication.processEvents()

        cap.release()
        out_video.release()

        df = pd.DataFrame(data)
        for keypoint in self.selected_keypoints:
            for axis in ['x', 'y', 'z', 'visibility']:
                col_name = f"{keypoint}_{axis}"
                df[col_name] = df[col_name].interpolate(method='linear', limit_direction='both')

        df.to_csv(output_csv_path, index=False)
        self.processing_complete.emit(f"CSV: {output_csv_path}\nVideo: {output_video_path}")

    def draw_pose_with_angles(self, frame, keypoint_coords, angles):
        # Define connections for visualization
        connections = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
        ]

        # Filter connections based on user-selected keypoints
        selected_connections = [
            (p1, p2) for p1, p2 in connections
            if p1 in self.selected_keypoints and p2 in self.selected_keypoints
        ]

        # Draw connections
        for p1, p2 in selected_connections:
            if p1 in keypoint_coords and p2 in keypoint_coords:
                pt1 = tuple(map(int, keypoint_coords[p1]))
                pt2 = tuple(map(int, keypoint_coords[p2]))
                color = (255, 0, 0) if "right" in p1 and "right" in p2 else (0, 255, 0)  # Blue for right, Green for left
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw selected keypoints as circles
        for keypoint, coords in keypoint_coords.items():
            if keypoint in self.selected_keypoints:  # Draw only selected keypoints
                pt = tuple(map(int, coords))
                color = (255, 0, 0) if "right" in keypoint else (0, 255, 0)  # Blue for right, Green for left
                cv2.circle(frame, pt, 6, color, -1)

        # Draw angles only for joints in the selected keypoints
        for joint, angle in angles.items():
            if joint in self.selected_keypoints:  # Display angle only if the joint is selected
                x, y = map(int, keypoint_coords[joint])
                angle_text = f"{int(angle)}"
                cv2.putText(frame, angle_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



    def calculate_angles(self, keypoint_coords):
        def calculate_angle(a, b, c):
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            dot_product = ba[0] * bc[0] + ba[1] * bc[1]
            magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
            magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

            if magnitude_ba == 0 or magnitude_bc == 0:
                return None

            cosine_angle = max(-1, min(1, dot_product / (magnitude_ba * magnitude_bc)))
            angle = math.acos(cosine_angle)
            return math.degrees(angle)

        # Define possible triplets for angle calculation
        triplets = [
            ("left_shoulder", "left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow", "right_wrist"),
            ("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle"),
            ("left_shoulder", "left_hip", "left_knee"),
            ("right_shoulder", "right_hip", "right_knee"),
        ]

        # Filter triplets based on `selected_keypoints`
        filtered_triplets = [
            (a, b, c) for a, b, c in triplets
            if a in self.selected_keypoints and b in self.selected_keypoints and c in self.selected_keypoints
        ]

        angles = {}
        for a, b, c in filtered_triplets:
            if a in keypoint_coords and b in keypoint_coords and c in keypoint_coords:
                angle = calculate_angle(keypoint_coords[a], keypoint_coords[b], keypoint_coords[c])
                if angle is not None:
                    angles[b] = round(angle, 2)  # Store angle for the middle joint
        return angles


class PoseEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Estimation Application")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel: Controls and Table for Angles
        self.left_panel = QVBoxLayout()
        self.init_controls_panel()
        self.init_angles_table()  # Initialize angles table
        self.init_speeds_table()  # Initialize speeds table

        # Right Panel: Video Display
        self.right_panel = QVBoxLayout()
        self.init_video_display()

        # Add panels to the main layout
        self.main_layout.addLayout(self.left_panel, 2)
        self.main_layout.addLayout(self.right_panel, 3)

        # Variables
        self.video_path = None
        self.output_dir = None
        self.workout_configurations = {}
        self.selected_workout = None
        self.pose_thread = None

        # Load workouts from database
        self.load_workouts_from_database()

        self.init_menu()

    def init_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_video_action = QAction("Open Video", self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)

    def init_controls_panel(self):
        # Group Box for Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Workout Buttons
        self.register_workout_button = QPushButton("Register Workouts", self)
        self.register_workout_button.clicked.connect(self.open_workout_registration)
        self.manage_workouts_button = QPushButton("Manage Workouts", self)
        self.manage_workouts_button.clicked.connect(self.open_manage_workouts_dialog)
        self.select_workout_button = QPushButton("Select Workout", self)
        self.select_workout_button.clicked.connect(self.open_workout_selection)
        self.start_button = QPushButton("Start Pose Estimation", self)
        self.start_button.clicked.connect(self.start_pose_estimation)

        # Add Buttons to Layout
        controls_layout.addWidget(self.register_workout_button)
        controls_layout.addWidget(self.manage_workouts_button)
        controls_layout.addWidget(self.select_workout_button)
        controls_layout.addWidget(self.start_button)

        # Add Controls Group to Left Panel
        self.left_panel.addWidget(controls_group)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.left_panel.addWidget(self.progress_bar)

    def init_angles_table(self):
        angles_group = QGroupBox("Angles")
        angles_layout = QVBoxLayout(angles_group)

        self.angles_table = QTableWidget()
        self.angles_table.setColumnCount(2)
        self.angles_table.setHorizontalHeaderLabels(["Joint", "Angle (°)"])
        self.angles_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        angles_layout.addWidget(self.angles_table)

        self.left_panel.addWidget(angles_group)
    
    def init_speeds_table(self):
        speeds_group = QGroupBox("Speeds")
        speeds_layout = QVBoxLayout(speeds_group)

        self.speeds_table = QTableWidget()
        self.speeds_table.setColumnCount(2)
        self.speeds_table.setHorizontalHeaderLabels(["Point", "Speed (cm/s)"])
        self.speeds_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        speeds_layout.addWidget(self.speeds_table)

        self.left_panel.addWidget(speeds_group)

    def init_video_display(self):
        # Group Box for Video Display
        video_group = QGroupBox("Video Display")
        video_layout = QVBoxLayout(video_group)

        # Video Display Area
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #ccc; background-color: #000;")
        video_layout.addWidget(self.video_label)

        # Add Video Group to Right Panel
        self.right_panel.addWidget(video_group)

    def load_workouts_from_database(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT name, keypoints FROM workouts')
        for name, keypoints in cursor.fetchall():
            self.workout_configurations[name] = keypoints.split(',')
        conn.close()

    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.video_path = file_path

    def open_workout_registration(self):
        dialog = WorkoutConfigurationDialog(self)
        dialog.exec_()
        self.workout_configurations.update(dialog.get_workout_configurations())

    def open_manage_workouts_dialog(self):
        dialog = ManageWorkoutsDialog(self)
        dialog.exec_()
        self.load_workouts_from_database()

    def open_workout_selection(self):
        if not self.workout_configurations:
            QMessageBox.warning(self, "Error", "No workouts have been registered.")
            return

        workout_names = list(self.workout_configurations.keys())
        selected_workout, ok = QInputDialog.getItem(self, "Select Workout", "Choose a workout:", workout_names, 0, False)
        if ok and selected_workout:
            self.selected_workout = selected_workout

    def start_pose_estimation(self):
        if not self.selected_workout:
            QMessageBox.warning(self, "Error", "Please select a workout for processing.")
            return

        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not self.output_dir:
            return

        workout_keypoints = self.workout_configurations[self.selected_workout]

        self.pose_thread = PoseEstimationThread(
            self.video_path, self.output_dir, workout_keypoints
        )
        self.pose_thread.frame_processed.connect(self.update_video_frame)
        self.pose_thread.angles_calculated.connect(self.update_angles_table)
        self.pose_thread.speeds_calculated.connect(self.update_speeds_table)
        self.pose_thread.processing_complete.connect(self.on_processing_complete)
        self.pose_thread.progress_updated.connect(self.progress_bar.setValue)
        self.pose_thread.start()

    def update_video_frame(self, frame):
        pixmap = QPixmap.fromImage(frame)
        self.video_label.setPixmap(pixmap)

    def update_angles_table(self, angles):
        self.angles_table.setRowCount(len(angles))
        for row, (joint, angle) in enumerate(angles.items()):
            self.angles_table.setItem(row, 0, QTableWidgetItem(joint))
            self.angles_table.setItem(row, 1, QTableWidgetItem(f"{angle}°"))

    def update_speeds_table(self, speeds):
        for point, speed in speeds.items():
            # Check if the point already exists in the table
            existing_rows = [self.speeds_table.item(row, 0).text() for row in range(self.speeds_table.rowCount())]
            
            if point in existing_rows:
                # Update the speed value in the table
                row_idx = existing_rows.index(point)
                self.speeds_table.setItem(row_idx, 1, QTableWidgetItem(f"{speed:.2f} cm/s"))
            else:
                # Add a new row if the point is not already in the table
                new_row = self.speeds_table.rowCount()
                self.speeds_table.insertRow(new_row)
                self.speeds_table.setItem(new_row, 0, QTableWidgetItem(point))
                self.speeds_table.setItem(new_row, 1, QTableWidgetItem(f"{speed:.2f} cm/s"))



    def on_processing_complete(self, output_paths):
        self.start_button.setText("Processing Complete! Start another video?")
        self.statusBar().showMessage(f"Files saved at: {output_paths}")


class WorkoutConfigurationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workout Configuration")
        self.setGeometry(200, 200, 600, 400)

        self.workout_configurations = {}

        layout = QVBoxLayout(self)

        self.workout_name_label = QLabel("Workout Name:")
        self.workout_name_input = QLineEdit(self)
        layout.addWidget(self.workout_name_label)
        layout.addWidget(self.workout_name_input)

        self.keypoints = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
            "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
            "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
            "left_heel", "right_heel", "left_foot_index", "right_foot_index"
        ]
        self.keypoint_checkboxes = []

        self.keypoints_label = QLabel("Select Keypoints for the Workout:")
        layout.addWidget(self.keypoints_label)

        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for idx, keypoint in enumerate(self.keypoints):
            checkbox = QCheckBox(keypoint, self)
            self.keypoint_checkboxes.append(checkbox)
            scroll_layout.addWidget(checkbox, idx // 3, idx % 3)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        self.add_workout_button = QPushButton("Add Workout", self)
        self.add_workout_button.clicked.connect(self.add_workout)
        layout.addWidget(self.add_workout_button)

        self.done_button = QPushButton("Done", self)
        self.done_button.clicked.connect(self.accept)
        layout.addWidget(self.done_button)

    def add_workout(self):
        workout_name = self.workout_name_input.text().strip()
        if not workout_name:
            QMessageBox.warning(self, "Error", "Workout name cannot be empty.")
            return

        selected_keypoints = [cb.text() for cb in self.keypoint_checkboxes if cb.isChecked()]
        if not selected_keypoints:
            QMessageBox.warning(self, "Error", "At least one keypoint must be selected.")
            return

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO workouts (name, keypoints)
                VALUES (?, ?)
            ''', (workout_name, ','.join(selected_keypoints)))
            conn.commit()
            conn.close()

            self.workout_configurations[workout_name] = selected_keypoints
            self.workout_name_input.clear()
            for cb in self.keypoint_checkboxes:
                cb.setChecked(False)

            QMessageBox.information(self, "Success", f"Workout '{workout_name}' added successfully.")
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Error", f"Workout '{workout_name}' already exists.")

    def get_workout_configurations(self):
        return self.workout_configurations


class ManageWorkoutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Workouts")
        self.setGeometry(200, 200, 600, 400)

        self.layout = QVBoxLayout(self)

        self.workouts_table = QTableWidget(self)
        self.workouts_table.setColumnCount(2)
        self.workouts_table.setHorizontalHeaderLabels(["Workout Name", "Keypoints"])
        self.workouts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layout.addWidget(self.workouts_table)

        self.load_workouts()

        self.delete_button = QPushButton("Delete Selected Workout", self)
        self.delete_button.clicked.connect(self.delete_workout)
        self.layout.addWidget(self.delete_button)

        self.edit_button = QPushButton("Edit Selected Workout", self)
        self.edit_button.clicked.connect(self.edit_workout)
        self.layout.addWidget(self.edit_button)

    def load_workouts(self):
        self.workouts_table.setRowCount(0)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT name, keypoints FROM workouts')
        for row_idx, (name, keypoints) in enumerate(cursor.fetchall()):
            self.workouts_table.insertRow(row_idx)
            self.workouts_table.setItem(row_idx, 0, QTableWidgetItem(name))
            self.workouts_table.setItem(row_idx, 1, QTableWidgetItem(keypoints))
        conn.close()

    def delete_workout(self):
        selected_row = self.workouts_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "Error", "No workout selected.")
            return

        workout_name = self.workouts_table.item(selected_row, 0).text()
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the workout '{workout_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM workouts WHERE name = ?', (workout_name,))
            conn.commit()
            conn.close()
            self.load_workouts()

    def edit_workout(self):
        selected_row = self.workouts_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "Error", "No workout selected.")
            return

        workout_name = self.workouts_table.item(selected_row, 0).text()
        keypoints = self.workouts_table.item(selected_row, 1).text().split(',')

        dialog = WorkoutConfigurationDialog(self)
        dialog.workout_name_input.setText(workout_name)
        for cb in dialog.keypoint_checkboxes:
            cb.setChecked(cb.text() in keypoints)

        if dialog.exec_() == QDialog.Accepted:
            updated_workout_name = dialog.workout_name_input.text().strip()
            updated_keypoints = ','.join([cb.text() for cb in dialog.keypoint_checkboxes if cb.isChecked()])

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE workouts
                SET name = ?, keypoints = ?
                WHERE name = ?
            ''', (updated_workout_name, updated_keypoints, workout_name))
            conn.commit()
            conn.close()
            self.load_workouts()


if __name__ == "__main__":
    initialize_database()
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    window = PoseEstimationApp()
    window.show()
    sys.exit(app.exec_())