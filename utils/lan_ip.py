"""Detect the LAN IP address reachable by other devices on the same network.

Supports a manual override via the ``SPLAZMATTE_LAN_IP`` environment variable
for machines where auto-detection picks up a virtual adapter address.

Run standalone to diagnose which IPs are detected::

    python -m utils.lan_ip
"""

import ipaddress
import logging
import os
import platform
import socket
import subprocess

log = logging.getLogger(__name__)

_RFC1918 = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)


def _is_rfc1918(ip: str) -> bool:
    """Return True if *ip* belongs to a private (RFC 1918) range."""
    return any(ipaddress.IPv4Address(ip) in net for net in _RFC1918)


def _default_gateway() -> str | None:
    """Return the default gateway IP, or None if it cannot be determined.

    On Windows, VPN/proxy tools (Clash, Surge, etc.) hijack the UDP
    default route so ``connect(('8.8.8.8', 80))`` returns the tunnel IP
    instead of the real LAN adapter.  We use PowerShell to read the
    routing table and find the real gateway.

    On macOS/Linux the UDP method already works, so we skip this.
    """
    if platform.system() != "Windows":
        return None
    try:
        r = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-NetRoute -DestinationPrefix 0.0.0.0/0"
                " | Sort-Object RouteMetric"
                " | Select-Object -First 1 -ExpandProperty NextHop",
            ],
            capture_output=True, text=True, timeout=10,
        )
        gw = r.stdout.strip()
        if gw and r.returncode == 0:
            return gw
    except Exception:
        log.debug("Failed to get default gateway via PowerShell", exc_info=True)
    return None


def detect_all_ips() -> list[tuple[str, str]]:
    """Return all detected IPv4 addresses with their source label.

    Returns:
        List of (ip, source) tuples, e.g. [("192.168.31.5", "gateway")].
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(ip: str, source: str) -> None:
        if ip not in seen:
            results.append((ip, source))
            seen.add(ip)

    # Method 1 (Windows): UDP connect to the real gateway from the
    # routing table.  This avoids VPN/proxy tunnel IPs.
    gw = _default_gateway()
    if gw:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((gw, 80))
                _add(s.getsockname()[0], "gateway")
        except Exception:
            pass

    # Method 2: UDP socket default route via 8.8.8.8
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            _add(s.getsockname()[0], "udp-route")
    except Exception:
        pass

    # Method 3: hostname resolution (may include virtual adapters)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            _add(info[4][0], "hostname")
    except Exception:
        pass

    return results


def lan_ip() -> str:
    """Return the best LAN IP for other devices to reach this machine.

    Check order:
      1. ``SPLAZMATTE_LAN_IP`` env var (manual override)
      2. Auto-detected RFC 1918 address (gateway > udp-route > hostname)
      3. First detected address regardless of range
      4. ``127.0.0.1`` as last resort
    """
    override = os.environ.get("SPLAZMATTE_LAN_IP", "").strip()
    if override:
        return override

    candidates = detect_all_ips()
    for ip, _source in candidates:
        if _is_rfc1918(ip):
            return ip
    return candidates[0][0] if candidates else "127.0.0.1"


if __name__ == "__main__":
    candidates = detect_all_ips()

    override = os.environ.get("SPLAZMATTE_LAN_IP", "").strip()
    if override:
        print(f"  env SPLAZMATTE_LAN_IP = {override}")
    else:
        print("  env SPLAZMATTE_LAN_IP = (not set)")

    print()
    if not candidates:
        print("  (no IPs detected)")
    else:
        selected = lan_ip()
        for ip, source in candidates:
            tag = " <-- selected" if ip == selected and not override else ""
            private = "private" if _is_rfc1918(ip) else "public"
            print(f"  {ip:<18}  source={source:<12}  {private}{tag}")

    print()
    print(f"  lan_ip() -> {lan_ip()}")
