#!/usr/bin/env python3
"""
YOLOX Detection Service using the simplified base framework with multi-GPU support.
Reads RTSP frames, runs YOLOX detection, publishes bounding boxes to MQTT.
"""

from typing import Any, Dict


from contanos.base_worker import BaseWorker
from rtmlib.tools.object_detection import YOLOX
class YOLOXWorker(BaseWorker):
    """YOLOX detection processor with multi-GPU parallel processing."""
    
    def __init__(self, worker_id: int, device: str, 
                 model_config: Dict,
                 input_interface, 
                 output_interface):
        super().__init__(worker_id, device, model_config,
                         input_interface, output_interface)
    
    def _model_init(self):
        self.model = YOLOX(**self.model_config,
                           device=self.device)  # Use the specific device for this model
        
    def _predict(self, input: Any, metadata: Any) -> Any:

        bboxes, det_scores = self.model(input)
        return {'scale': 1, 'bboxes': bboxes, 'det_scores': det_scores, 'classes': [-1] * len(det_scores)}
