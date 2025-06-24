#!/usr/bin/env python3
"""
RTMPose service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from rtmpose_worker import RTMPoseWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.io.mqtt_input_interface import MQTTInput
from contanos.io.multi_input_interface import MultiInputInterface
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string
from contanos.utils.yaml_config_loader import ConfigLoader

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose RTMPose for Pose Estimation"
    )
    
    # Add config file argument
    parser.add_argument('--config', type=str, default='pose_estimation_config.yaml',
                       help='Path to YAML configuration file')

    # Optional overrides (these will override YAML config if provided)
    add_argument(parser, 'in_rtsp', 'in_rtsp', None)
    add_argument(parser, 'in_mqtt', 'in_mqtt', None)
    add_argument(parser, 'out_mqtt', 'out_mqtt', None)
    add_argument(parser, 'devices', 'devices', None)
    add_argument(parser, 'model_input_size', 'model_input_size', None, arg_type=int, nargs=2)

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    # Load YAML configuration
    config_loader = ConfigLoader(args.config)
    rtmpose_config = config_loader.get_service_config('rtmpose')
    
    # Get multi-input configuration
    multi_inputs = config_loader.get_multi_input_config_strings('rtmpose')
    
    # Get configuration values (CLI args override YAML)
    if args.in_rtsp:
        in_rtsp = args.in_rtsp
    else:
        in_rtsp = multi_inputs.get('rtsp', config_loader.get_input_config_string('rtmpose', 'rtsp'))
    
    if args.in_mqtt:
        in_mqtt = args.in_mqtt
    else:
        in_mqtt = multi_inputs.get('mqtt', config_loader.get_input_config_string('rtmpose', 'mqtt'))
    
    out_mqtt = args.out_mqtt if args.out_mqtt else config_loader.get_output_config_string('rtmpose')
    devices = args.devices if args.devices else config_loader.get_devices('rtmpose')
    log_level = args.log_level if hasattr(args, 'log_level') and args.log_level else config_loader.get_log_level()
    backend = args.backend if hasattr(args, 'backend') and args.backend else config_loader.get_backend('rtmpose')
    
    # Model input size from YAML or CLI
    if args.model_input_size:
        model_input_size = args.model_input_size
    else:
        model_input_size = rtmpose_config.get('model_input_size', [192, 256])
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting RTMPose service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  in_mqtt: {in_mqtt}")
    logger.info(f"  out_mqtt: {out_mqtt}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  model_input_size: {model_input_size}")
    logger.info(f"  backend: {backend}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        in_mqtt_config = parse_config_string(in_mqtt)
        out_mqtt_config = parse_config_string(out_mqtt)

        # Create input/output interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface = MQTTInput(config=in_mqtt_config)
        input_interface = MultiInputInterface([input_video_interface, input_message_interface])
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()
        
        # Create model configuration
        model_config = dict(
            onnx_model=rtmpose_config.get('model_url',
                'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'),
            model_input_size=model_input_size,
            backend=backend,
        )

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=RTMPoseWorker,
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
        
        logger.info("RTMPose service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting RTMPose service: {e}")
        raise
    finally:
        logger.info("RTMPose service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        rtsp_q = input_interface.interfaces[0].queue.qsize()  # RTSP queue
        mqtt_q = input_interface.interfaces[1].message_queue.qsize()  # MQTT queue
        
        logging.info(f"Main Q: {main_q}, Sync Dict: {sync_dict}, RTSP Q: {rtsp_q}, MQTT Q: {mqtt_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 