import socket
import urllib.request
import httpx
import requests

urls = [
    "https://gamma-api.polymarket.com/markets?closed=false&active=true&limit=5&search=Bitcoin",
    "https://external-api.kalshi.com/trade-api/v2/markets?limit=5&status=open&series_ticker=BTC"
]

print("--- 1. Testing DNS resolution (gethostbyname) ---")
for url in urls:
    host = url.split("//")[1].split("/")[0].split("?")[0]
    try:
        ip = socket.gethostbyname(host)
        print(f"DNS OK: {host} -> {ip}")
    except Exception as e:
        print(f"DNS FAILED: {host} -> {e}")

print("\n--- 2. Testing with urllib (Standard Python Library) ---")
for url in urls:
    host = url.split("//")[1].split("/")[0].split("?")[0]
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            print(f"urllib OK: {host} (Status: {response.status})")
    except Exception as e:
        print(f"urllib FAILED: {host} -> {e}")

print("\n--- 3. Testing with requests ---")
for url in urls:
    host = url.split("//")[1].split("/")[0].split("?")[0]
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        print(f"requests OK: {host} (Status: {r.status_code})")
    except Exception as e:
        print(f"requests FAILED: {host} -> {e}")

print("\n--- 4. Testing with httpx ---")
for url in urls:
    host = url.split("//")[1].split("/")[0].split("?")[0]
    try:
        r = httpx.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        print(f"httpx OK: {host} (Status: {r.status_code})")
    except Exception as e:
        print(f"httpx FAILED: {host} -> {e}")
