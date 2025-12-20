#!/usr/bin/env bash
#
# =============================================================================
#  WireGuard + AI side‑car (ML / RL / Federated Learning) demo
#
#  What it creates:
#   • Docker container running WireGuard (linuxserver/wireguard)
#   • Docker container running a tiny Python inference engine (wg-ai-detector)
#   • iptables NFQUEUE rule that sends every packet traversing wg0 to the
#     Python side‑car for AI‑based allow/block decisions.
#
#  Prerequisites:
#   • Linux kernel with WireGuard support (>= v1.0)
#   • Docker Engine (>= 20.10) + root / sudo rights
#
#  NOTE: This script is **educational** – you will still need to:
#        • Generate WireGuard keys and configure peers
#        • Train / replace the ML model (ddos_detector.pt)
#        • Implement your RL / FL logic inside detector.py
# =============================================================================

set -euo pipefail

# ------------------------------- Config ---------------------------------------
# You can export these before invoking the script to override defaults.
WG_SERVER_URL="${WG_SERVER_URL:-vpn.example.com}"   # Public DNS / IP
WG_SERVER_PORT="${WG_SERVER_PORT:-51820}"           # UDP port
WG_PEERS="${WG_PEERS:-1}"                          # How many peers to init
WG_INTERNAL_SUBNET="${WG_INTERNAL_SUBNET:-10.13.13.0}" # wg subnet
WG_SUBNET_MASK="${WG_SUBNET_MASK:-24}"              # CIDR mask
WG_DNS="${WG_DNS:-auto}"                           # DNS for clients

# Docker names
DOCKER_NET="wgnet"
WG_CONF_VOLUME="wgconf"
AI_IMAGE_NAME="wg-ai-detector"
WG_CONTAINER_NAME="wireguard"
AI_CONTAINER_NAME="wg-ai-detector"

# Where we will generate the side‑car sources
SIDE_CAR_DIR="${PWD}/ml-sidecar"

# ---------------------------------------------------------------------------

# ------------------------------- Helper functions ----------------------------
log() {
    echo -e "\e[32m[+] $*\e[0m"
}
err() {
    echo -e "\e[31m[!] $*\e[0m" >&2
}
die() {
    err "$*"
    exit 1
}
# ---------------------------------------------------------------------------

# -------------------------- Prerequisite checks ----------------------------
if [[ $EUID -ne 0 ]]; then
    die "This script must be run as root (or via sudo)."
fi

if ! command -v docker >/dev/null 2>&1; then
    die "Docker not found – please install Docker Engine first."
fi

if ! docker info >/dev/null 2>&1; then
    die "Docker daemon is not running or you lack permissions."
fi

# ---------------------------------------------------------------------------

# -------------------------- Prepare AI side‑car ----------------------------
log "Creating side‑car source directory at ${SIDE_CAR_DIR}"
mkdir -p "${SIDE_CAR_DIR}"

# ---- Dockerfile -----------------------------------------------------------
cat > "${SIDE_CAR_DIR}/Dockerfile" <<'EOF'
# -------------------------------------------------
#   wg‑ai‑detector – tiny Python inference engine
# -------------------------------------------------
FROM python:3.11-slim

# System deps (netfilter queue + iptables helpers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        iptables iproute2 \
        libnetfilter-queue-dev \
        gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages (adjust versions as needed)
RUN pip install --no-cache-dir \
        torch==2.2.0 \
        torchvision \
        scapy \
        netfilterqueue \
        flask \
        numpy \
        pandas \
        scikit-learn \
        fedml \
        stable-baselines3

# Warm‑up the torch cache (optional, speeds first import)
RUN python -c "import torch; print('Torch version:', torch.__version__)"

WORKDIR /app
COPY detector.py .
# If you have a pre‑trained model, copy it to /models/ (see below)
COPY models/ /models/

EXPOSE 8080
CMD ["python", "detector.py"]
EOF

# ---- Placeholder model ----------------------------------------------------
# Create empty folder for model files – replace with your real *.pt later.
mkdir -p "${SIDE_CAR_DIR}/models"
# Touch a dummy file so the folder is not empty (Docker COPY needs something).
touch "${SIDE_CAR_DIR}/models/README.md"

# ---- detector.py ------------------------------------------------------------
cat > "${SIDE_CAR_DIR}/detector.py" <<'PYTHON'
#!/usr/bin/env python3
"""
wg‑ai‑detector – minimal inference engine with hooks for
* reinforcement learning (RL) policies
* federated learning (FL) aggregation
* DDoS / zero‑day packet classification

The script uses NetfilterQueue (NFQUEUE) to receive packets from the kernel,
runs a tiny Torch model, and decides whether to ACCEPT or DROP each packet.
A tiny Flask API is also exposed on port 8080 for FL coordination (optional).

You *must* replace the placeholder model (ddos_detector.pt) with a real one,
and you can extend the `rl_agent` / `fedml_client` objects with your own logic.
"""

import os
import sys
import json
import ipaddress
import logging
from pathlib import Path

# -------------------- Core imports --------------------
import torch
import numpy as np
import scapy.all as scapy
from netfilterqueue import NetfilterQueue

# Optional Flask API (used for FL, RL‑policy updates, etc.)
from flask import Flask, request, jsonify
from threading import Thread

# -------------------- FedML (Federated Learning) --------------------
# Placeholder – you would set up a FedML client that talks to your
# aggregation server.  For a real deployment replace the stub below.
# from fedml import FedMLClient

# -------------------- RL agent (e.g. Stable‑Baselines3) --------------------
# Placeholder – you can load a policy net, update it online, etc.
# from stable_baselines3 import PPO

# -------------------- Configuration --------------------
MODEL_PATH = Path("/models/ddos_detector.pt")
MODEL = None   # will be loaded lazily
DEVICE = torch.device("cpu")   # change to "cuda" if you have a GPU

# Simple Flask app that can be called by other federated peers
app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# -------------------- Helper functions --------------------
def load_model():
    """Load (or reload) the Torch model from disk."""
    global MODEL
    if not MODEL_PATH.is_file():
        logging.warning(f"Model file {MODEL_PATH} not found – using dummy model.")
        # Dummy model – random output
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return torch.rand(1)
        MODEL = Dummy()
    else:
        MODEL = torch.load(MODEL_PATH, map_location=DEVICE)
        MODEL.eval()
    logging.info("ML model loaded.")


def ip_to_int(ip_str):
    """Convert dotted‑quad IP string to integer (feature engineering)."""
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except ipaddress.AddressValueError:
        return 0


def extract_features(pkt):
    """
    Very naïve feature extractor – you will replace this with something
    richer (e.g., flow‑based stats, packet‑size distribution, time‑window
    aggregates, etc.).
    """
    # Ensure we have an IP layer – otherwise return zeros
    if not pkt.haslayer(scapy.IP):
        return [0, 0, 0, 0]

    ip = pkt[scapy.IP]
    src_ip = ip_to_int(ip.src)
    dst_ip = ip_to_int(ip.dst)
    proto = ip.proto
    length = ip.len

    # Example: Append TCP/UDP ports if present
    src_port = dst_port = 0
    if pkt.haslayer(scapy.TCP):
        src_port = pkt[scapy.TCP].sport
        dst_port = pkt[scapy.TCP].dport
    elif pkt.haslayer(scapy.UDP):
        src_port = pkt[scapy.UDP].sport
        dst_port = pkt[scapy.UDP].dport

    return [src_ip, dst_ip, proto, length, src_port, dst_port]


def predict(packet):
    """
    Run the model on a scapy packet.
    Returns a probability (0‑1) that the packet is malicious.
    """
    features = extract_features(packet)
    # Convert to tensor (float32)
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = MODEL(x)
    # Our dummy model returns a scalar in [0,1]; for real nets adjust.
    if isinstance(out, torch.Tensor):
        prob = out.item()
    else:
        prob = float(out)
    return prob


# -------------------- NFQUEUE callback --------------------
def packet_callback(pkt):
    """netfilterqueue callback – decide accept/drop."""
    try:
        payload = pkt.get_payload()
        scapy_pkt = scapy.IP(payload)
    except Exception as e:
        logging.error(f"Failed to parse packet: {e}")
        pkt.accept()
        return

    # Run inference
    prob = predict(scapy_pkt)

    # Simple threshold – tune as needed (e.g., based on ROC analysis)
    THRESHOLD = float(os.getenv("AI_DROP_THRESHOLD", "0.85"))

    if prob >= THRESHOLD:
        logging.info(f"[BLOCKED] prob={prob:.3f}  {scapy_pkt.summary()}")
        pkt.drop()
    else:
        # Include optional RL policy check here
        pkt.accept()


# -------------------- Flask API (optional) --------------------
@app.route("/model/upload", methods=["POST"])
def upload_model():
    """
    A simple endpoint that allows a federated peer (or CI) to replace the model.
    In a real FL setup you would also verify signatures, aggregate updates,
    etc.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    file = request.files["file"]
    dest = MODEL_PATH
    file.save(str(dest))
    load_model()                 # reload the new model immediately
    return jsonify({"status": "model updated"}), 200


@app.route("/healthz")
def health():
    return jsonify({"status": "alive"}), 200


def run_flask():
    """Run Flask in a background thread (non‑blocking)."""
    # Bind to all interfaces – the side‑car container lives in the same
    # network namespace as WireGuard, but exposing it on 0.0.0.0 keeps it reachable.
    app.run(host="0.0.0.0", port=8080)


# -------------------- Main entry point --------------------
def main():
    # Load (or create) the model
    load_model()

    # Start Flask in separate thread (optional – comment out if you don't need it)
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logging.info("Flask API listening on :8080 (for FL / model update)")

    # Set up NFQUEUE
    nfqueue = NetfilterQueue()
    QUEUE_NUM = int(os.getenv("AI_NFQUEUE_NUM", "0"))
    logging.info(f"Binding to NFQUEUE #{QUEUE_NUM}")
    nfqueue.bind(QUEUE_NUM, packet_callback)

    try:
        logging.info("Entering NFQUEUE event loop – press Ctrl‑C to stop")
        nfqueue.run()
    except KeyboardInterrupt:
        logging.info("Interrupted – shutting down")
    finally:
        nfqueue.unbind()


if __name__ == "__main__":
    main()
PYTHON

# ------------------- Build the AI image ------------------------------------
log "Building AI side‑car image (${AI_IMAGE_NAME})"
docker build -t "${AI_IMAGE_NAME}" "${SIDE_CAR_DIR}"

# ---------------------------------------------------------------------------

# ------------------- Prepare WireGuard ------------------------------------
log "Pulling latest WireGuard Docker image"
docker pull linuxserver/wireguard

log "Creating Docker network '${DOCKER_NET}' (if not exists)"
docker network inspect "${DOCKER_NET}" >/dev/null 2>&1 || \
    docker network create "${DOCKER_NET}"

log "Ensuring config volume '${WG_CONF_VOLUME}' exists"
docker volume inspect "${WG_CONF_VOLUME}" >/dev/null 2>&1 || \
    docker volume create "${WG_CONF_VOLUME}"
# ---------------------------------------------------------------------------

# ------------------- Run WireGuard container -------------------------------
# NOTE: The linuxserver image auto‑generates keys when you first start it.
#       You can later mount the volume and edit the config files to add
#       peers, set DNS, etc.

log "Starting WireGuard container '${WG_CONTAINER_NAME}'"
docker rm -f "${WG_CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
    --name "${WG_CONTAINER_NAME}" \
    --restart unless-stopped \
    --cap-add NET_ADMIN --cap-add SYS_MODULE \
    -e PUID=$(id -u) -e PGID=$(id -g) \
    -e TZ=$(cat /etc/timezone 2>/dev/null || echo UTC) \
    -e SERVERURL="${WG_SERVER_URL}" \
    -e SERVERPORT="${WG_SERVER_PORT}" \
    -e PEERS="${WG_PEERS}" \
    -e INTERNAL_SUBNET="${WG_INTERNAL_SUBNET}" \
    -e INTERNAL_SUBNET_MASK="${WG_SUBNET_MASK}" \
    -e PEERDNS="${WG_DNS}" \
    -p "${WG_SERVER_PORT}:51820/udp" \
    -v "${WG_CONF_VOLUME}:/config" \
    --network "${DOCKER_NET}" \
    --sysctl="net.ipv4.ip_forward=1" \
    --sysctl="net.ipv6.conf.all.forwarding=1" \
    linuxserver/wireguard

# ---------------------------------------------------------------------------

# ------------------- Run AI side‑car container ----------------------------
# The container shares the network namespace of the WireGuard container
# (so it sees the wg0 interface) and also the PID namespace (optional,
# handy for reading /proc/net/dev etc.).  It needs NET_ADMIN to manipulate
# iptables if you later want it to self‑adjust policies.

log "Starting AI side‑car container '${AI_CONTAINER_NAME}'"
docker rm -f "${AI_CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
    --name "${AI_CONTAINER_NAME}" \
    --restart unless-stopped \
    --cap-add NET_ADMIN \
    --cap-add SYS_ADMIN \
    --pid "container:${WG_CONTAINER_NAME}" \
    --network "container:${WG_CONTAINER_NAME}" \
    -v "${SIDE_CAR_DIR}/models:/models:ro" \
    -e AI_NFQUEUE_NUM=0 \
    -e AI_DROP_THRESHOLD=0.85 \
    "${AI_IMAGE_NAME}"

# ---------------------------------------------------------------------------

# ------------------- Install NFQUEUE iptables rule -----------------------
# This rule sends *all* forwarded packets that cross the wg0 interface
# into NFQUEUE #0 where our Python side‑car will see them.

# First remove any old rule (ignore errors)
iptables -D FORWARD -i wg0 -j NFQUEUE --queue-num 0 2>/dev/null || true

log "Adding iptables rule to forward wg0 traffic to NFQUEUE #0"
iptables -I FORWARD -i wg0 -j NFQUEUE --queue-num 0

log "Enabling IP forwarding (if not already set)"
sysctl -w net.ipv4.ip_forward=1 >/dev/null

# ---------------------------------------------------------------------------

# ------------------- Final info -----------------------------------------
cat <<'EOS'

=====================================================================
✅ WireGuard VPN + AI side‑car is up and running!

🔹 WireGuard container name : ${WG_CONTAINER_NAME}
🔹 AI detector container   : ${AI_CONTAINER_NAME}
🔹 Listening UDP port     : ${WG_SERVER_PORT}
🔹 IP forwarding enabled : yes

📦 Next steps (you will probably want to do all of these):

 1️⃣  **Generate / edit WireGuard config**  
     The config volume is persisted in Docker volume "${WG_CONF_VOLUME}".
     You can inspect it with:
        docker run --rm -it -v ${WG_CONF_VOLUME}:/config alpine cat /config/wg0.conf
     Add your peers, public keys, allowed IPs, etc., then restart the
     container (`docker restart ${WG_CONTAINER_NAME}`).

 2️⃣  **Provide a real ML model**  
     Replace the placeholder file `${SIDE_CAR_DIR}/models/ddos_detector.pt`
     with a model trained on your own traffic (binary classifier, anomaly
     detector, etc.).  The demo loads the model automatically; you can
     also POST a new model to the Flask API:
        curl -X POST -F "file=@my_model.pt" http://<host_ip>:8080/model/upload

 3️⃣  **Plug in RL / FL logic**  
     • Reinforcement‑learning: instantiate a PPO/TD3 agent (stable‑baselines3) 
       inside `detector.py`, let it adjust the `AI_DROP_THRESHOLD` or even
       issue `iptables` rate‑limit commands based on reward signals.  
     • Federated learning: use FedML (or TensorFlow Federated) to periodically
       pull model updates from a central aggregator and call `load_model()`.  

 4️⃣  **Test the DDoS mitigation**  
     From a client (or using `hping3`), generate a flood of packets toward a
     service behind the VPN and watch the terminal output of the AI container:
        docker logs -f ${AI_CONTAINER_NAME}
     You should see "[BLOCKED]" lines when the model flags traffic as malicious.

 5️⃣  **Secure the containers**  
     • Rotate WireGuard keys regularly.  
     • Run the containers with a non‑root user inside (add `-e PUID=$(id -u)`).  
     • Harden the host firewall (limit inbound UDP 51820 to known IPs).  
     • Use TLS/HTTPS for any management API you expose (Flask can be behind
       a lightweight reverse‑proxy such as Caddy or Nginx).

🚀  Happy hacking!  When you’re ready for production, consider:

   • Running the AI side‑car on dedicated hardware (GPU) for lower latency.  
   • Using a proper model‑registry and CI/CD pipeline for safe model deployments.  
   • Integrating alerting (Prometheus + Alertmanager) for anomalies.  

If you run into trouble, the logs of each container are your best friend:

   docker logs ${WG_CONTAINER_NAME}
   docker logs ${AI_CONTAINER_NAME}

=====================================================================
EOS

exit 0
