"""
Simple example of receiving and sending messages via RabbitMQ.
"""

import os
import json
import logging
import signal
import sys
import subprocess
from dotenv import load_dotenv
import pika
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_PORT = int(os.getenv('RABBITMQ_PORT', 5672))
RABBITMQ_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'default')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_DEFAULT_PASS', 'default')
RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'default')
API_BASE_URL = os.getenv("API_BASE_URL", "default")
INPUT_QUEUE = os.getenv('INPUT_QUEUE', 'scraper_new_offers')
OUTPUT_QUEUE = os.getenv('OUTPUT_QUEUE', 'poi_results')


class ProcessingTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ProcessingTimeout()


class RabbitMQProcessor:
    """Simple RabbitMQ message processor."""

    def __init__(self):
        self.connection = None
        self.channel = None

    def connect(self):
        """Connect to RabbitMQ."""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.basic_qos(prefetch_count=1)

            # Declare queues (if they do not exist)
            self.channel.queue_declare(queue=INPUT_QUEUE, durable=True)
            self.channel.queue_declare(queue=OUTPUT_QUEUE, durable=True)

            logger.info(
                "Connected to RabbitMQ: %s:%s",
                RABBITMQ_HOST,
                RABBITMQ_PORT
            )
        except Exception as e:
            logger.error("RabbitMQ connection error: %s", e)
            raise

    def process_message(self, message_data: dict) -> dict:
        apartment_id = message_data.get("apartment_id")

        if apartment_id is None:
            logger.error(
                "Missing apartment_id (reason=missing field). input=%s",
                message_data
            )
            raise Exception("Missing apartment_id")

        try:
            apartment = self.api_get(f"/apartments/{int(apartment_id)}")
        except Exception as e:
            logger.error(
                "API error for apartment_id=%s (reason=%s)",
                apartment_id,
                e
            )
            raise

        geo = apartment.get("geolocation") or {}
        lat = geo.get("lat")
        lon = geo.get("lng")

        if lat is None or lon is None:
            logger.error(
                "API data error for apartment_id=%s (reason=%s)",
                apartment_id,
                "missing geolocation.lat/lng"
            )
            raise Exception("Apartment geolocation missing")

        script = os.getenv("AUTO_POI_SCRIPT")
        grid_json = os.getenv("GRID_JSON")
        workspace = os.getenv("WORKSPACE_DIR")
        poi_query = os.getenv("POI_QUERY_SCRIPT")

        cmd = [
            sys.executable, script,
            "--lat", str(lat),
            "--lon", str(lon),
            "--grid-json", grid_json,
            "--workspace", workspace,
            "--poi-query-script", poi_query,
        ]

        p = subprocess.run(cmd, text=True, capture_output=True)

        if p.returncode != 0:
            logger.error(
                "Algorithm error for apartment_id=%s (reason=%s). stderr_tail=%s",
                apartment_id,
                f"exit code {p.returncode}",
                (p.stderr or "")[-800:]
            )
            raise Exception(f"Algorithm error exit_code={p.returncode}")

        try:
            result_json = json.loads(p.stdout)
        except Exception as e:
            logger.error(
                "Algorithm error for apartment_id=%s (reason=%s)",
                apartment_id,
                e
            )
            raise Exception("Algorithm stdout invalid JSON") from e

        pois_list = result_json.get("pois_in_range") or result_json.get("pois") or []

        created = 0
        linked = 0
        errors = []
        poi_cache = {}

        for it in pois_list:
            try:
                category = it["category"]
                geo = it["geolocation"]

                if isinstance(geo, dict):
                    lat_p = float(geo["lat"])
                    lng_p = float(geo["lng"])
                elif isinstance(geo, (list, tuple)) and len(geo) >= 2:
                    lat_p = float(geo[0])
                    lng_p = float(geo[1])
                else:
                    raise ValueError(f"Unsupported geolocation format: {geo}")

                time_to_poi = int(it["time_to_poi"])
                key = (category, lat_p, lng_p)

                if key not in poi_cache:
                    poi_req = {
                        "category": category,
                        "geolocation": {"lat": lat_p, "lng": lng_p}
                    }
                    poi_resp = self.api_post("/pois", poi_req)

                    if isinstance(poi_resp, list):
                        if not poi_resp or not isinstance(poi_resp[0], dict):
                            raise RuntimeError(
                                f"POST /pois returned unexpected list: {poi_resp}"
                            )
                        poi_resp = poi_resp[0]

                    if not isinstance(poi_resp, dict):
                        raise RuntimeError(
                            f"POST /pois returned unexpected type "
                            f"{type(poi_resp)}: {poi_resp}"
                        )

                    poi_id = poi_resp.get("id")
                    if poi_id is None:
                        raise RuntimeError(
                            f"POST /pois did not return id: {poi_resp}"
                        )

                    poi_cache[key] = int(poi_id)
                    created += 1

                link_path = f"/apartments/{int(apartment_id)}/pois"
                link_req = {
                    "poi_id": poi_cache[key],
                    "time_to_poi": time_to_poi
                }
                self.api_post(link_path, link_req)
                linked += 1

            except Exception as e:
                errors.append({"item": it, "error": str(e)})

        if errors:
            logger.error(
                "POI save errors for apartment_id=%s (reason=%s)",
                apartment_id,
                f"{errors} errors"
            )
            raise Exception(f"POI save failed ({errors} errors)")

        logger.info("Processed OK for apartment_id=%s", apartment_id)
        return {"apartment_id": apartment_id}

    def api_post(self, path: str, payload: dict) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    def api_get(self, path: str, params: dict | None = None) -> dict:
        url = f"{API_BASE_URL}{path}"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    def on_message(self, ch, method, properties, body):
        """Handle received message."""
        apartment_id = None
        try:
            # Parse JSON message
            message_data = json.loads(body.decode('utf-8'))
            apartment_id = message_data.get("apartment_id")

            logger.info("Received message: %s", message_data)

            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(25 * 60)

            try:
                processed_data = self.process_message(message_data)
            finally:
                signal.alarm(0)

            ch.basic_publish(
                exchange='',
                routing_key=OUTPUT_QUEUE,
                body=json.dumps(processed_data),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # persistent message
                )
            )

            # Acknowledge message processing
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(
                "Message sent to queue: %s",
                OUTPUT_QUEUE
            )

        except ProcessingTimeout:
            logger.warning(
                "TIMEOUT processing apartment_id=%s after %ss -> SKIP (ACK). body_tail=%s",
                apartment_id,
                25 * 60,
                body[:500]
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON format: %s", e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except Exception as e:
            logger.error("Processing error: %s", e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start(self):
        """Start listening for messages."""
        try:
            self.channel.basic_consume(
                queue=INPUT_QUEUE,
                on_message_callback=self.on_message
            )

            logger.info(
                "Waiting for messages from queue '%s'. CTRL+C to exit",
                INPUT_QUEUE
            )
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping...")
            self.stop()

    def stop(self):
        """Stop processing and close the connection."""
        if self.channel and not self.channel.is_closed:
            self.channel.stop_consuming()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        logger.info("Connection closed")


def signal_handler(signum, frame):
    """Handle termination signal."""
    logger.info("Termination signal received")
    sys.exit(0)


def main():
    """Main application entry point."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    processor = RabbitMQProcessor()

    try:
        processor.connect()
        processor.start()
    except Exception as e:
        logger.error("Application error: %s", e)
        sys.exit(1)
    finally:
        processor.stop()


if __name__ == '__main__':
    main()
