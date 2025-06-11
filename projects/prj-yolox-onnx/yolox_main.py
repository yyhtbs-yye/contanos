#!/usr/bin/env python3
"""
Main service template with environment variable configuration.
"""
import os
import asyncio
import logging
import argparse

# Import your modules here
from yolox_worker import YOLOXWorker
from contanos.io.rtsp_input_interface import RTSPInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose YOLOX for Bounding Box Detection"
    )
    add_argument(parser, 'in_rtsp', 'in_rtsp', 'rtsp://192.168.200.206:8554,topic=mystream')
    add_argument(parser, 'out_mqtt', 'out_mqtt', 'mqtt://192.168.200.206:1883,topic=yolox,qos=2,queue_max_len=50')
    add_argument(parser, 'devices', 'devices', 'cuda:1')
    add_argument(parser, 'model_input_size', 'model_input_size', [640, 640], arg_type=int, nargs=2)

    add_service_args(parser)
    add_compute_args(parser)
    
    return parser.parse_args()


async def main():
    """Main function to create and start the service."""
    # Parse environment configuration
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting service with configuration:")
    for key, value in vars(args).items():  # Fixed: use vars(args) instead of args.items()
        logger.info(f"  {key}: {value}")
    
    try:
        in_rtsp_config = parse_config_string(args.in_rtsp)
        out_mqtt_config = parse_config_string(args.out_mqtt)
        
        # Create input/output interfaces
        input_interface = RTSPInput(config=in_rtsp_config)
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration
        model_config = dict(onnx_model=('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
                                        'yolox_m_8xb8-300e_humanart-c2c7a14a.zip'),
                            model_input_size=args.model_input_size,
                            backend=args.backend,
        )

        # Convert devices string to list if needed
        devices = args.devices.split(',') if isinstance(args.devices, str) else args.devices

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=YOLOXWorker,
            model_config=model_config,
            devices=devices,                    # Fixed: use processed devices list
            input_interface=input_interface,
            output_interface=output_interface,
            num_workers_per_device=args.num_workers_per_device,
        )
        
        # Start the service
        service = await start_a_service(
            processor=processor,
            run_until_complete=args.run_until_complete,
            daemon_mode=args.daemon_mode,
        )
        
        logger.info("Service started successfully")
        
        if args.daemon_mode:
            logger.info("Running in daemon mode - service will continue in background")
            # Keep the main thread alive if needed
            while True:
                await asyncio.sleep(60)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error starting service: {e}")
        raise
    finally:
        logger.info("Service shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())