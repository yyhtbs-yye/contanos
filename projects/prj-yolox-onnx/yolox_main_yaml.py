#!/usr/bin/env python3
"""
YOLOX service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from yolox_worker import YOLOXWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string
from contanos.utils.yaml_config_loader import ConfigLoader

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose YOLOX for Bounding Box Detection"
    )
    
    # Add config file argument
    parser.add_argument('--config', type=str, default='pose_estimation_config.yaml',
                       help='Path to YAML configuration file')
    
    # Optional overrides (these will override YAML config if provided)
    add_argument(parser, 'in_rtsp', 'in_rtsp', None)
    add_argument(parser, 'out_mqtt', 'out_mqtt', None)
    add_argument(parser, 'devices', 'devices', None)
    add_argument(parser, 'model_input_size', 'model_input_size', None, arg_type=int, nargs=2)

    add_service_args(parser)
    add_compute_args(parser)
    
    return parser.parse_args()

async def main():
    """Main function to create and start the service."""
    args = parse_args()
    
    # Load YAML configuration
    config_loader = ConfigLoader(args.config)
    yolox_config = config_loader.get_service_config('yolox')
    
    # Get configuration values (CLI args override YAML)
    in_rtsp = args.in_rtsp if args.in_rtsp else config_loader.get_input_config_string('yolox')
    out_mqtt = args.out_mqtt if args.out_mqtt else config_loader.get_output_config_string('yolox')
    devices = args.devices if args.devices else config_loader.get_devices('yolox')
    log_level = args.log_level if hasattr(args, 'log_level') and args.log_level else config_loader.get_log_level()
    backend = args.backend if hasattr(args, 'backend') and args.backend else config_loader.get_backend('yolox')
    
    # Model input size from YAML or CLI
    if args.model_input_size:
        model_input_size = args.model_input_size
    else:
        model_input_size = yolox_config.get('model_input_size', [640, 640])
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting YOLOX service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  out_mqtt: {out_mqtt}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  model_input_size: {model_input_size}")
    logger.info(f"  backend: {backend}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        out_mqtt_config = parse_config_string(out_mqtt)
        
        # Create input/output interfaces
        input_interface = RTSPInput(config=in_rtsp_config)
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration
        model_config = dict(
            onnx_model=yolox_config.get('model_url', 
                'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'),
            model_input_size=model_input_size,
            backend=backend,
        )

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]
        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=YOLOXWorker,
            model_config=model_config,
            devices=devices,
            input_interface=input_interface,
            output_interface=output_interface,
            num_workers_per_device=args.num_workers_per_device,
        )
        
        # Start the service
        service = await start_a_service(
            processor=processor,
            run_until_complete=args.run_until_complete,
            daemon_mode=False,
        )
        
        logger.info("YOLOX service started successfully")
        
 
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting YOLOX service: {e}")
        raise
    finally:
        logger.info("YOLOX service shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 