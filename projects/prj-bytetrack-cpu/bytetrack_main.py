#!/usr/bin/env python3
"""
Main service template with environment variable configuration.
"""
import os
import asyncio
import logging
import argparse

# Import your modules here
from bytetrack_worker import ByteTrackWorker
from contanos.io.mqtt_sorted_input_interface import MQTTSortedInput
from contanos.io.mqtt_output_interface import MQTTOutput
from contanos.helpers.create_a_processor import create_a_processor
from contanos.helpers.start_a_service import start_a_service
from contanos.utils.create_args import add_argument, add_service_args, add_compute_args
from contanos.utils.setup_logging import setup_logging
from contanos.utils.parse_config_string import parse_config_string

def parse_args():
    parser = argparse.ArgumentParser(
        description="BoxMOTLite ByteTrack Detection for Object Tracking"
    )
    add_argument(parser, 'in_mqtt', 'in_mqtt', 'mqtt://192.168.200.206:1883,topic=yolox,qos=2,buffer_threshold=100')
    add_argument(parser, 'out_mqtt', 'out_mqtt', 'mqtt://192.168.200.206:1883,topic=bytetrack,qos=2,queue_max_len=100')
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
        in_mqtt_config = parse_config_string(args.in_mqtt)
        out_mqtt_config = parse_config_string(args.out_mqtt)

        # Create input/output interfaces
        input_interface = MQTTSortedInput(config=in_mqtt_config)
        output_interface = MQTTOutput(config=out_mqtt_config)
        
        await input_interface.initialize()
        await output_interface.initialize()

        # Create model configuration
        model_config = dict(min_conf=0.1, track_thresh=0.45,
                            match_thresh=0.8, track_buffer=25,
                            frame_rate=30)

        monitor_task = asyncio.create_task(quick_debug())

        # Convert devices string to list if needed
        devices = ['cpu']

        # Create processor with workers
        _, processor = create_a_processor(
            worker_class=ByteTrackWorker,
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
        message_q = input_interface.ordered_queue.qsize()  # MQTT queue
        ordered_q = input_interface.ordered_queue.qsize()  # MQTT queue
        
        print(f"MESSAGE Q: {message_q}, ORDERED Q: {ordered_q}, ")
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())