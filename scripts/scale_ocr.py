import argparse
import logging
import subprocess
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("icp")
 
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Scale the OCR deployment and update its variant, cores, and batching.")
    parser.add_argument("-v", "--variant", type=str, required=True, help="Variant to run (required)")
    parser.add_argument("-r", "--replicas", type=int, default=1, help="Number of replicas (default 1)")
    parser.add_argument("-c", "--cores", type=int, default=1, help="CPU cores per replica (default 1)")
    parser.add_argument("-m", "--max_batch", type=int, default=1, help="Max batch size (default 1)")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Batch timeout in ms (default 10)")
    return parser.parse_args()
 
def config_scale(replicas, cores, variant, max_batch, timeout):
    log.info("Scaling begun")
    subprocess.run(["kubectl", "set", "env", "deployment/ocr", f"VARIANT={variant}", f"MAX_BATCH={max_batch}", f"BATCH_TIMEOUT_MS={timeout}"], check=True)
    subprocess.run(["kubectl", "set", "resources", "deployment/ocr", f"--requests=cpu={cores}", f"--limits=cpu={cores}"], check=True)
    subprocess.run(["kubectl", "scale", "deployment/ocr", f"--replicas={replicas}"], check=True)
    subprocess.run(["kubectl", "rollout", "status", "deployment/ocr", "--timeout=300s"], check=True)
    log.info("Scaled to %s replicas x %s cores, variant=%s, max_batch=%s, timeout=%sms", replicas, cores, variant, max_batch, timeout)
 
if __name__ == "__main__":
    args = parse_arguments()
    config_scale(args.replicas, args.cores, args.variant, args.max_batch, args.timeout)