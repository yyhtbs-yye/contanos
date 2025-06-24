#!/usr/bin/env python3
"""
Annotation service with YAML configuration support.
"""
import os
import sys
import asyncio
import logging
import argparse

# Add parent directories to path for contanos imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import your modules here
from annotation_worker import AnnotationWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.rtsp_output_interface import RTSPOutput
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
        description="Annotation Visualization Service"
    )
    
    # Add config file argument
    parser.add_argument('--config', type=str, default='pose_estimation_config.yaml',
                       help='Path to YAML configuration file')

    # Optional overrides (these will override YAML config if provided)
    add_argument(parser, 'in_rtsp', 'in_rtsp', None)
    add_argument(parser, 'in_mqtt1', 'in_mqtt1', None)
    add_argument(parser, 'in_mqtt2', 'in_mqtt2', None)
    add_argument(parser, 'out_rtsp', 'out_rtsp', None)
    add_argument(parser, 'devices', 'devices', None)

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()

async def main():
    global input_interface
    """Main function to create and start the service."""
    args = parse_args()
    
    # Load YAML configuration
    config_loader = ConfigLoader(args.config)
    annotation_config = config_loader.get_service_config('annotation')
    
    # Get multi-input configuration
    multi_inputs = config_loader.get_multi_input_config_strings('annotation')
    
    # Get configuration values (CLI args override YAML)
    if args.in_rtsp:
        in_rtsp = args.in_rtsp
    else:
        in_rtsp = multi_inputs.get('rtsp', config_loader.get_input_config_string('annotation', 'rtsp'))
    
    if args.in_mqtt1:
        in_mqtt1 = args.in_mqtt1
    else:
        in_mqtt1 = multi_inputs.get('mqtt1', config_loader.get_input_config_string('annotation', 'mqtt1'))
    
    if args.in_mqtt2:
        in_mqtt2 = args.in_mqtt2
    else:
        in_mqtt2 = multi_inputs.get('mqtt2', config_loader.get_input_config_string('annotation', 'mqtt2'))
    
    out_rtsp = args.out_rtsp if args.out_rtsp else config_loader.get_output_config_string('annotation')
    devices = args.devices if args.devices else config_loader.get_devices('annotation')
    log_level = args.log_level if hasattr(args, 'log_level') and args.log_level else config_loader.get_log_level()
    
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Annotation service with configuration:")
    logger.info(f"  in_rtsp: {in_rtsp}")
    logger.info(f"  in_mqtt1: {in_mqtt1}")
    logger.info(f"  in_mqtt2: {in_mqtt2}")
    logger.info(f"  out_rtsp: {out_rtsp}")
    logger.info(f"  devices: {devices}")
    logger.info(f"  log_level: {log_level}")
    
    try:
        in_rtsp_config = parse_config_string(in_rtsp)
        in_mqtt1_config = parse_config_string(in_mqtt1)
        in_mqtt2_config = parse_config_string(in_mqtt2)
        out_rtsp_config = parse_config_string(out_rtsp)

        # Create input interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface1 = MQTTInput(config=in_mqtt1_config)
        input_message_interface2 = MQTTInput(config=in_mqtt2_config)
        input_interface = MultiInputInterface([input_video_interface, input_message_interface1, input_message_interface2])
        
        # Initialize input interface first
        await input_interface.initialize()
        output_interface = RTSPOutput(config=out_rtsp_config)
        await output_interface.initialize()
        
        # Create model configuration
        model_config = dict()

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = devices.split(',') if isinstance(devices, str) else [devices]

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=AnnotationWorker,
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
        
        logger.info("Annotation service started successfully")
        
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting Annotation service: {e}")
        raise
    finally:
        logger.info("Annotation service shutdown complete")

# Debug monitoring function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        rtsp_q = input_interface.interfaces[0].queue.qsize()  # RTSP queue
        mqtt1_q = input_interface.interfaces[1].message_queue.qsize()  # MQTT queue
        mqtt2_q = input_interface.interfaces[2].message_queue.qsize()  # MQTT queue
        logging.info(f"Main Q: {main_q}, Sync Dict: {sync_dict}, RTSP Q: {rtsp_q}, MQTT1 Q: {mqtt1_q}, MQTT2 Q: {mqtt2_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main()) 