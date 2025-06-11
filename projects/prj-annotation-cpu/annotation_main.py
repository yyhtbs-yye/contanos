#!/usr/bin/env python3
"""
Main service template with environment variable configuration.
"""
import os
import asyncio
import logging
import argparse

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenMMPose RTMPose for Pose Estimation"
    )

    add_argument(parser, 'in_rtsp', 'in_rtsp', 'rtsp://192.168.200.206:8554,topic=mystream')
    add_argument(parser, 'in_mqtt1', 'in_mqtt1', 'mqtt://192.168.200.206:1883,topic=bytetrack,qos=2,queue_max_len=100')
    add_argument(parser, 'in_mqtt2', 'in_mqtt2', 'mqtt://192.168.200.206:1883,topic=rtmpose,qos=2,queue_max_len=100')
    add_argument(parser, 'out_rtsp', 'out_rtsp', 'rtsp://192.168.200.206:8554,topic=outstream,width=1920,height=1080,fps=25')
    add_argument(parser, 'devices', 'devices', 'cpu')

    add_service_args(parser)
    add_compute_args(parser)

    return parser.parse_args()


async def main():

    global input_interface
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
        in_mqtt1_config = parse_config_string(args.in_mqtt1)
        in_mqtt2_config = parse_config_string(args.in_mqtt2)
        out_rtsp_config = parse_config_string(args.out_rtsp)

        # Create input/output interfaces
        input_video_interface = RTSPInput(config=in_rtsp_config)
        input_message_interface1 = MQTTInput(config=in_mqtt1_config)
        input_message_interface2 = MQTTInput(config=in_mqtt2_config)
        input_interface = MultiInputInterface([input_video_interface, input_message_interface1, input_message_interface2])
        output_interface = RTSPOutput(config=out_rtsp_config)
        
        await input_interface.initialize()
        await output_interface.initialize()
        
        # Create model configuration
        model_config = dict()

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = args.devices.split(',') if isinstance(args.devices, str) else args.devices

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=AnnotationWorker,
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

# Add this to your main function
async def quick_debug():
    while True:
        main_q = input_interface._queue.qsize()
        sync_dict = len(input_interface._data_dict)
        
        rtsp_q = input_interface.interfaces[0].queue.qsize()  # RTSP queue
        mqtt1_q = input_interface.interfaces[1].message_queue.qsize()  # MQTT queue
        mqtt2_q = input_interface.interfaces[2].message_queue.qsize()  # MQTT queue
        
        print(f"Main Q: {main_q}, Sync Dict: {sync_dict}, RTSP Q: {rtsp_q}, MQTT1 Q: {mqtt1_q}, MQTT2 Q: {mqtt2_q}")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
    