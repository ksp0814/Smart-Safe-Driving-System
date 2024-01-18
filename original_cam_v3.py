import cv2, time
import numpy as np
import logging
import pycuda.driver as drv



import torch
from matplotlib import pyplot as plt
import dlib
import winsound
from math import hypot
import threading
import time

from taskConditions import TaskConditions, Logger
from ObjectDetector.yoloDetector import YoloDetector
from ObjectDetector.utils import ObjectModelType,  CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from TrafficLaneDetector.ultrafastLaneDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ultrafastLaneDetector.utils import LaneModelType, OffsetType, CurvatureType

temp2 = []   #오른쪽 temp2
temp2_left = [] 	#왼쪽 temp2
fps2 = 0
temp_eye = []
temp_eye_left = []
temp_eye_right = []

lane_change = 999
eye_change = 999

#
def lane():
	global lane_change
	lane_change = 999
	font = cv2.FONT_HERSHEY_PLAIN
	def show_image_right():
		# cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
		img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-left_lanes.png')
		text = 'Wake Up!!!'
		color = (0, 0, 255)
		cv2.putText(img, text, (50, 50), font, 3, color, 2, cv2.LINE_AA)
		cv2.imshow('Patrick', img)
		cv2.waitKey(3000)
		cv2.destroyWindow('Patrick')
	
	def show_image_left():
		# cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
		img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-right_lanes.png')
		text = 'Wake Up!!!'
		color = (0, 0, 255)
		cv2.putText(img, text, (50, 50), font, 3, color, 2, cv2.LINE_AA)
		cv2.imshow('Patrick', img)
		cv2.waitKey(3000)
		cv2.destroyWindow('Patrick')

	

	def play_beep2():
		frequency2 = 2500
		duration2 = 1000
		winsound.Beep(frequency2, duration2)
		

	t5 = threading.Thread(target=play_beep2)

	t3 = threading.Thread(target=lane)
	LOGGER = Logger(None, logging.INFO, logging.INFO )
	def stop():
		t4.join()

	def stop2():
		t5.join()

	video_path = "./TrafficLaneDetector/temp/test_01.mp4"
    
	lane_config = {
		"model_path": "./TrafficLaneDetector/models/culane_res18.trt",
		"model_type" : LaneModelType.UFLDV2_CULANE
	}

	object_config = {
		"model_path": './ObjectDetector/models/yolov8m.trt',
		"model_type" : ObjectModelType.YOLOV8,
		"classes_path" : './ObjectDetector/models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45
	}

	# Priority : FCWS > LDWS > LKAS
	class ControlPanel(object):
		CollisionDict = {
							CollisionType.UNKNOWN : (0, 255, 255),
							CollisionType.NORMAL : (0, 255, 0),
							CollisionType.PROMPT : (0, 102, 255),
							CollisionType.WARNING : (0, 0, 255)
						}

		OffsetDict = { 
						OffsetType.UNKNOWN : (0, 255, 255), 
						OffsetType.RIGHT :  (0, 0, 255), 
						OffsetType.LEFT : (0, 0, 255), 
						OffsetType.CENTER : (0, 255, 0)
					}

		CurvatureDict = { 
							CurvatureType.UNKNOWN : (0, 255, 255),
							CurvatureType.STRAIGHT : (0, 255, 0),
							CurvatureType.EASY_LEFT : (0, 102, 255),
							CurvatureType.EASY_RIGHT : (0, 102, 255),
							CurvatureType.HARD_LEFT : (0, 0, 255),
							CurvatureType.HARD_RIGHT : (0, 0, 255)
						}

		def __init__(self):
			collision_warning_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
			self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
			collision_prompt_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
			self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
			collision_normal_img = cv2.imread('./TrafficLaneDetector/trafficSigns/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
			self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
			left_curve_img = cv2.imread('./TrafficLaneDetector/trafficSigns/left_turn.png', cv2.IMREAD_UNCHANGED)
			self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
			right_curve_img = cv2.imread('./TrafficLaneDetector/trafficSigns/right_turn.png', cv2.IMREAD_UNCHANGED)
			self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
			keep_straight_img = cv2.imread('./TrafficLaneDetector/trafficSigns/straight.png', cv2.IMREAD_UNCHANGED)
			self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
			determined_img = cv2.imread('./TrafficLaneDetector/trafficSigns/warn.png', cv2.IMREAD_UNCHANGED)
			self.determined_img = cv2.resize(determined_img, (200, 200))
			left_lanes_img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
			self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
			right_lanes_img = cv2.imread('./TrafficLaneDetector/trafficSigns/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
			self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))

			

			# FPS
			self.fps = 0
			self.frame_count = 0
			self.start = time.time()

			self.curve_status = None

		def _updateFPS(self) :
			"""
			Update FPS.

			Args:
				None

			Returns:
				None
			"""
			self.frame_count += 1
			if self.frame_count >= 30:
				self.end = time.time()
				self.fps = self.frame_count / (self.end - self.start)
				self.frame_count = 0
				self.start = time.time()

		def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
			"""
			Display Signs Panel on image.

			Args:
				main_show: image.
				offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
				curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

			Returns:
				main_show: Draw sings info on frame.
			"""


			W = 400
			H = 365
			widget = np.copy(main_show[:H, :W])
			widget //= 2
			# widget[0:3,:] = [0, 0, 255]  # top
			# widget[-3:-1,:] = [0, 0, 255] # bottom
			# widget[:,0:3] = [0, 0, 255]  #left
			# widget[:,-3:-1] = [0, 0, 255] # right
			widget[0:3,:] = 0  # top
			widget[-3:-1,:] = 0 # bottom
			widget[:,0:3] = 0  #left
			widget[:,-3:-1] = 0 # right
			main_show[:H, :W] = widget

			global fps2
			
			global temp

			while True:
				if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
					y, x = self.determined_img[:,:,3].nonzero()
					main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
					self.curve_status = None

				elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
					(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
					y, x = self.left_curve_img[:,:,3].nonzero()
					main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
					self.curve_status = "Left"

				elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
					(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
					y, x = self.right_curve_img[:,:,3].nonzero()
					main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
					self.curve_status = "Right"
				
				
				t10 = threading.Thread(target=show_image_right)
				t20 = threading.Thread(target=show_image_left)
			
				
				if ( offset_type == OffsetType.RIGHT ) :   # 이게 왼쪽 차선 물엇을때
					global lane_change
					# print("왼쪽")
					global temp2_left
					lane_change = 0
					temp2_left = [] * len(temp2_left)
					y, x = self.left_lanes_img[:,:,2].nonzero()
					main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
					# print("OffsetType.RIGHT")
					temp = 1000000000
					global temp2
					temp2.append(min(temp, fps2))

					temp_min = temp2[0]

					if fps2 > temp_min + 72 :   #   24 가 1초임 기억해
						print("sound")
						winsound.Beep(1000,1000)
						t10.start()
						
						fps2 = 0
												
						temp2 = [] * len(temp2)
					print("왼쪽 : ",lane_change)
						
					# if time.time() - prev_time2 > 3 :
					# 	# if not t5.is_alive():
					# 	# 	t5.start()
					# 		# winsound.Beep(1000,1000)
					# 		t5.start()
					# 		print("sound")
					# 		prev_time2 = time.time()
			
				elif ( offset_type == OffsetType.LEFT ) :  # 이게 오른쪽 차선 물었을때
					# print("오른쪽")
					lane_change = 1
					temp2 = [] * len(temp2)
					y, x = self.right_lanes_img[:,:,2].nonzero()
					main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
					# print("OffsetType.LEFT")
					temp = 1000000000
					
					temp2_left.append(min(temp, fps2))

					# print("temp",temp2_left[0],' fps2_right : ', fps2)

					temp_min_left = temp2_left[0]

					if fps2 > temp_min_left + 72 :   #24가 1초야 !!!
						print("sound")
						winsound.Beep(1000,1000)
						t20.start()
						
						# if not t5.is_alive():
						# 	t5.start()
							
						# else:
						# 	print("t5 thread is already running")
					
						fps2 = 0
						
						
						temp2_left = [] * len(temp2_left)
					print("오른쪽 : ",lane_change)
					# winsound.Beep(1000,1000)
					
					
					# if time.time() - prev_time2 > 3 :
					# 	t5.start()
					# 	print("siu~~~")
					# 	prev_time2 = 0
				# elif offset_type == OffsetType.LEFT and time.time() - prev_time_left > 1:
				# 	t5.start()
				# 	winsound.Beep(1000,1000)
				# 	print("right sound")
				# 	prev_time_left = 0
				
				elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
					lane_change = 999
					y, x = self.keep_straight_img[:,:,3].nonzero()
					main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
					self.curve_status = "Straight"

					
				self._updateFPS()
				cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
				cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
				cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
				return main_show

		def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
			"""
			Display Collision Panel on image.

			Args:
				main_show: image.
				collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
				obect_infer_time: object detection time -> float.
				lane_infer_time:  lane detection time -> float.

			Returns:
				main_show: Draw collision info on frame.
			"""

			W = int(main_show.shape[1]* show_ratio)
			H = int(main_show.shape[0]* show_ratio)

			widget = np.copy(main_show[H+20:2*H, -W-20:])
			widget //= 2
			widget[0:3,:] = 0  # top
			widget[-3:-1,:] = 0 # bottom
			widget[:,-3:-1] = 0  #left
			widget[:,0:3] = 0 # right
			main_show[H+20:2*H, -W-20:] = widget

			if (collision_type == CollisionType.WARNING) :
				y, x = self.collision_warning_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]

			elif (collision_type == CollisionType.PROMPT) :
				y, x =self.collision_prompt_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]

			elif (collision_type == CollisionType.NORMAL) :
				y, x = self.collision_normal_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

			cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
			cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
			cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
			return main_show


	if __name__ == "__main__":

		# Initialize read and save video 
		cap = cv2.VideoCapture(video_path)
		fps_lane = cap.get(cv2.CAP_PROP_FPS)
		width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
		cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)	
		
		#==========================================================
		# 					Initialize Class
		#==========================================================
		LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
		LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))

		# lane detection model
		LOGGER.info("UfldDetector Model Type : {}".format(lane_config["model_type"].name))
		if ( "UFLDV2" in lane_config["model_type"].name) :
			UltrafastLaneDetectorV2.set_defaults(lane_config)
			laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
		else :
			UltrafastLaneDetector.set_defaults(lane_config)
			laneDetector = UltrafastLaneDetector(logger=LOGGER)
		transformView = PerspectiveTransformation( (width, height) )

		# object detection model
		LOGGER.info("YoloDetector Model Type : {}".format(object_config["model_type"].name))
		YoloDetector.set_defaults(object_config)
		objectDetector = YoloDetector(LOGGER)
		distanceDetector = SingleCamDistanceMeasure()

		# display panel
		displayPanel = ControlPanel()
		
		analyzeMsg = TaskConditions()

		global fps2
		while cap.isOpened():
			
			ret, frame = cap.read() # Read frame from the video
			if ret:	
				frame_show = frame.copy()
				fps2 += 1

				#========================= Detect Model ========================
				obect_time = time.time()
				objectDetector.DetectFrame(frame)
				obect_infer_time = round(time.time() - obect_time, 2)
				lane_time = time.time()
				laneDetector.DetectFrame(frame)
				lane_infer_time = round(time.time() - lane_time, 4)

				#========================= Analyze Status ========================
				distanceDetector.calcDistance(objectDetector.object_info)
				vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.draw_area_points)

				analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.draw_area)

				if (not laneDetector.draw_area or analyzeMsg.CheckStatus()) :
					lanes_list = list(laneDetector.lanes_points)
					transformView.updateParams(lanes_list[1], lanes_list[2], analyzeMsg.transform_status)
				top_view_show = transformView.forward(frame_show)

				# if (not laneDetector.draw_area or analyzeMsg.CheckStatus()) :
				# 	transformView.updateParams(laneDetector.lanes_points[1], laneDetector.lanes_points[2], analyzeMsg.transform_status)
				# top_view_show = transformView.forward(frame_show)

				adjust_lanes_points = []
				for lanes_point in laneDetector.lanes_points :
					adjust_lanes_point = transformView.transformPoints(lanes_point)
					adjust_lanes_points.append(adjust_lanes_point)

				(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(top_view_show, adjust_lanes_points[1], adjust_lanes_points[2])

				analyzeMsg.UpdateOffsetStatus(vehicle_offset)
				analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

				#========================== Draw Results =========================
				transformView.DrawDetectedOnFrame(top_view_show, adjust_lanes_points, analyzeMsg.offset_msg)
				laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
				objectDetector.DrawDetectedOnFrame(frame_show)
				distanceDetector.DrawDetectedOnFrame(frame_show)

				frame_show = laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
				frame_show = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)	
				frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )
				frame_show = transformView.DisplayBirdView(frame_show, top_view_show)
				cv2.imshow("ADAS Simulation", frame_show)

			else:
				break
			vout.write(frame_show)	
			if cv2.waitKey(1) == ord('q'): 		# Press key q to stop
				stop()
				break

		vout.release()
		cap.release()
		cv2.destroyAllWindows()
	
def face_cam():
	global eye_change
	eye_change = 999
	fps2_eye = 0
	t4 = threading.Thread(target=face_cam)
	video_cam = "./TrafficLaneDetector/temp/cam1007.mp4"
	model = torch.hub.load('ultralytics/yolov5', 'custom', path='face6/yolov5/runs/train/exp17/weights/last.pt', force_reload=True)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	font = cv2.FONT_HERSHEY_PLAIN

	is_eye_closed = False

	drowsy_detected = False
	drowsy_start_time = None
	drowsy_duration = 0.0
	filename = 'C:\\Users\\kang\\cu\\Vehicle-CV-ADAS\\TrafficLaneDetector\\temp\\video_cam.avi'

	cam = cv2.VideoCapture(video_cam)
	fps2_eye = cam.get(cv2.CAP_PROP_FPS)
	width_cam  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	height_cam = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


	# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vout_cam = cv2.VideoWriter(filename, fourcc , 30.0, (width_cam, height_cam))


	def show_image():
		# cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
		img = cv2.imread('image.jpg')
		text = 'Wake Up!!!'
		color = (0, 0, 255)
		cv2.putText(img, text, (50, 50), font, 3, color, 2, cv2.LINE_AA)
		cv2.imshow('Patrick', img)
		cv2.waitKey(3000)
		cv2.destroyWindow('Patrick')

	def play_beep():
		frequency = 2500
		duration = 1000
		winsound.Beep(frequency, duration)

	def midpoint(p1, p2):
		return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

	def get_blinking_ratio(eye_points, facial_landmarks):
		left_point = facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y
		right_point = facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y
		center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
		center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

		hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
		var_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

		ratio = hor_line_length / var_line_length
		return ratio

	def get_gaze_ratio(eye_points, facial_landmarks):
		left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
									(facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
									(facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
									(facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
									(facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
									(facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

		height, width, _ = frame_cam.shape
		mask = np.zeros((height, width), np.uint8)
		cv2.polylines(frame_cam, [left_eye_region], True, 255, 2)
		cv2.fillPoly(mask, [left_eye_region], 255)
		eye = cv2.bitwise_and(gray, gray, mask=mask)

		min_x = np.min(left_eye_region[:, 0])
		max_x = np.max(left_eye_region[:, 0])
		min_y = np.min(left_eye_region[:, 1])
		max_y = np.max(left_eye_region[:, 1])

		gray_eye = eye[min_y: max_y, min_x: max_x]
		_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
		height, width = threshold_eye.shape
		left_side_thrshold = threshold_eye[0: height, 0: int(width / 2)]
		left_side_white = cv2.countNonZero(left_side_thrshold)
		right_side_thrshold = threshold_eye[0: height, int(width / 2): width]
		right_side_white = cv2.countNonZero(right_side_thrshold)

		if left_side_white == 0:
			gaze_ratio = 1
		elif right_side_white == 0:
			gaze_ratio = 5
		else:
			gaze_ratio = left_side_white / right_side_white
		return gaze_ratio
	def stop():
		t4.join()

	if __name__ == "__main__":

		while cam.isOpened():		
			ret, frame2 = cam.read()
			frame_cam = frame2.copy()
			if ret:
				# frame = cv2.flip(frame, 1)
				fps2_eye += 1
				cv2.VideoCapture(video_cam)
				gray = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2GRAY)


				t1 = threading.Thread(target=show_image)
				t2 = threading.Thread(target=play_beep)
				faces = detector(gray)
				for face in faces:
					landmarks = predictor(gray, face)

					left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
					right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47, 48], landmarks)
					blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                #----------------------------------------------------# 세필이가 만진 구간
					if blinking_ratio > 5.7:
						cv2.putText(frame_cam, "Blink", (250, 200), font, 2, (0, 0, 255), 3)
						global temp_eye
						print("blink")
						temp = 100000000
						temp_eye.append(min(temp, fps2_eye))

						# print('temp_eye : ', temp_eye[0],' fps2_eye : ', fps2_eye)

						temp_eye_min = temp_eye[0]

						if fps2_eye > temp_eye_min + 90 :
							print("sound")
							t1.start()

							fps2_eye = 0

							temp_eye = [] * len(temp_eye)

						if not is_eye_closed:
							is_eye_closed = True
							# prev_time = time.time()
					else:
						temp_eye = [] * len(temp_eye)
						is_eye_closed = False
				#-------------------------------------------------------# 세필이가 만진 구간

					gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
					gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47, 48], landmarks)
					gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

					if gaze_ratio <= 0.7 :  # 왼쪽을 보고 있을 때
						eye_change = 0
						global temp_eye_right
						temp_eye_right = [] * len(temp_eye_right)
						cv2.putText(frame_cam, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
						global temp_eye_left
						
						temp = 100000000
						temp_eye_left.append(min(temp, fps2_eye))
					
						temp_eye_left_min = temp_eye_left[0]
						print("오른쪽 왼쪽 : ",eye_change)

						# print('temp_eye_LEFT : ', temp_eye_left[0],' fps2_eye_LEFT : ', fps2_eye)

						if fps2_eye > temp_eye_left_min + 90 :
							print("sound")
							t1.start()

							fps2_eye = 0

							temp_eye_left = [] * len(temp_eye_left)

						global lane_change
						# print("확인용", lane_change)

						if lane_change == 1:
							print("======================눈왼쪽 차선오른 lane + eye")
							t1.start()


					elif 0.7 < gaze_ratio < 2: # 중앙을 보고 있을 때
						eye_change = 999
						temp_eye_right = [] * len(temp_eye_right)
						temp_eye_left = [] * len(temp_eye_left)						
						cv2.putText(frame_cam, "CENTER", (250, 100), font, 2, (0, 0, 255), 3)

					elif gaze_ratio >= 2: # 오른쪽을 보고 있을 때
						eye_change = 1
						temp_eye_left = [] * len(temp_eye_left)												
						cv2.putText(frame_cam, "RIGHT", (500, 100), font, 2, (0, 0, 255), 3)					
						temp = 100000000
						temp_eye_right.append(min(temp, fps2_eye))

						# print('temp_eye_RIGHT : ', temp_eye_right[0],' fps2_eye_RIGHT : ', fps2_eye)

						temp_eye_right_min = temp_eye_right[0]
						print("오른쪽 눈 : ",eye_change)

						if fps2_eye > temp_eye_right_min + 90 :
							print("sound")
							t1.start()

							fps2_eye = 0

							temp_eye_right = [] * len(temp_eye_right)
						
						
						# print("확인용", lane_change)

						if lane_change == 0:
							print("==============================눈오른 차선왼쪽 lane + eye")
							t1.start()

					# if lane_change == 0 and eye_change == 1:
					# 	# (lane_change == 1 and eye_change == 0) or
					# 	#왼쪽 0 오른쪽 1
					# 	print("lane + eye")
					# 	print("lane + eye")
					# 	print("lane + eye")
					# 	print("lane + eye")
					# 	print("lane + eye")
					# 	t1.start()

				results = model(frame_cam)

				for obj in results.xyxy[0]:
					if obj[-1] == 16:  # 16 corresponds to the "drowsy" class
						if not drowsy_detected:
							drowsy_detected = True
							drowsy_start_time = time.monotonic()
						else:
							drowsy_duration = time.monotonic() - drowsy_start_time
							if drowsy_duration >= 3.0:
								t1.start()
								print("drowsy")
								# t2.start()
								drowsy_duration = 0.0  # reset the duration
								drowsy_detected = False  # reset the detection flag
								drowsy_start_time = None  # reset the start time
								break
					else:
						drowsy_detected = False
						drowsy_duration = 0.0
						drowsy_start_time = None
				frame_cams = np.squeeze(results.render())
				cv2.imshow('YOLO', frame_cams)
				vout_cam.write(frame_cams)
			else:
				break
				
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				stop()
				break

		
		vout_cam.release()		
		cam.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	t3 = threading.Thread(target=lane)
	t4 = threading.Thread(target=face_cam)

	t3.start()
	t4.start()



