#!/usr/bin/env python3
"""
YOLOX Detection Service using the simplified base framework with multi-GPU support.
Reads RTSP frames, runs YOLOX detection, publishes bounding boxes to MQTT.
"""

from typing import Any, Dict
import numpy as np

from contanos.visualizer.box_drawer import BoxDrawer
from contanos.visualizer.skeleton_drawer import SkeletonDrawer
from contanos.visualizer.trajectory_drawer import TrajectoryDrawer

from contanos.base_worker import BaseWorker

class AnnotationWorker(BaseWorker):
    """ByteTrack tracking processor with single CPU serial processing."""
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.box_drawer = BoxDrawer()  # Use the specific device for this model
        self.skeleton_drawer = SkeletonDrawer()  # Use the specific device for this model
        self.trajectory_drawer = TrajectoryDrawer()  # Use the specific device for this model
        
    def _predict(self, input: Any, metadata: Any) -> Any:

        frame = input[0]
        bboxes = input[1]['results']['bboxes']
        track_ids = [int(it) for it in input[1]['results']['track_ids']]
        track_scores = input[1]['results']['track_scores']
        scale = input[1]['results']['scale']
        keypoints = input[2]['results']['keypoints']


        self.box_drawer.draw_boxes(frame, 
                                   bboxes=bboxes,
                                   track_ids=track_ids,
                                   scores=track_scores,
                                   scale=scale)
        frame = self.skeleton_drawer.draw_keypoints(frame, 
                                            keypoints=keypoints,
                                            track_ids=[0.5 for i in range(len(keypoints))],
                                            scale=scale)
        # self.trajectory_drawer.draw_trajectories()

        return {'img': frame}

