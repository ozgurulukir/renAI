import json
import logging
import os
import time
import requests
import webbrowser
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Dict, Any
from pathlib import Path

from config import OAUTH_PROVIDERS, AUTH_DIR
from utils import ensure_secure_dir

TOKEN_FILE = AUTH_DIR / "tokens.json"

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == "/callback":
            query_components = urllib.parse.parse_qs(parsed_path.query)
            if "code" in query_components:
                self.server.oauth_code = query_components["code"][0]
                self.server.oauth_state = query_components.get("state", [""])[0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization successful!</h1><p>You can close this window and return to the application.</p></body></html>")
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authorization failed!</h1><p>No code found in the request.</p></body></html>")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging

class OAuthManager:
    def __init__(self) -> None:
        ensure_secure_dir(AUTH_DIR)

        # Make tokens.json secure if it doesn't exist
        if not TOKEN_FILE.exists():
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump({}, f)
            if os.name != "nt":
                os.chmod(TOKEN_FILE, 0o600)

    def _load_tokens(self) -> Dict[str, Any]:
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading tokens: {e}")
            return {}

    def _save_tokens(self, tokens: Dict[str, Any]) -> None:
        try:
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump(tokens, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving tokens: {e}")

    def get_token(self, provider: str) -> Optional[str]:
        tokens = self._load_tokens()
        provider_data = tokens.get(provider)
        if not provider_data:
            return None

        # Here we could implement token refresh logic if expires_at < time.time()
        # For simplicity, we assume the token is valid or we just return it.
        return provider_data.get("access_token")

    def perform_device_code_flow(self, provider: str, config: Dict[str, Any]) -> bool:
        """Perform the OAuth2 Device Code Flow for a selected provider."""
        client_id = config.get("client_id")
        try:
            resp = requests.post(
                config["device_authorization_url"],
                data={"client_id": client_id},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            device_code = data.get("device_code")
            user_code = data.get("user_code")
            verification_uri = data.get("verification_uri")
            interval = data.get("interval", 5)

            if not device_code or not user_code or not verification_uri:
                print("Failed to get device code parameters from provider.")
                return False

            print("\n" + "=" * 50)
            print("Action Required: OAuth Authorization (Device Flow)")
            print("=" * 50)
            print(f"1. A browser window will now open automatically to: \n   {verification_uri}")
            print(f"   (If it doesn't open, please click the link above.)")
            print(f"2. Enter this code to authenticate:\n   {user_code}")
            print("\nWaiting for authorization...")

            try:
                webbrowser.open(verification_uri)
            except Exception as e:
                logging.debug(f"Could not open browser automatically: {e}")

            start_time = time.time()
            timeout = data.get("expires_in", 300)

            while time.time() - start_time < timeout:
                token_resp = requests.post(
                    config["token_url"],
                    data={
                        "client_id": client_id,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code
                    },
                    timeout=10
                )

                if token_resp.status_code == 200:
                    token_data = token_resp.json()
                    access_token = token_data.get("access_token")
                    refresh_token = token_data.get("refresh_token")
                    expires_in = token_data.get("expires_in", 3600)

                    tokens = self._load_tokens()
                    tokens[provider] = {
                        "access_token": access_token,
                        "refresh_token": refresh_token,
                        "expires_at": time.time() + expires_in
                    }
                    self._save_tokens(tokens)
                    print("\nSuccessfully logged in!")
                    return True

                elif token_resp.status_code == 400:
                    error_data = token_resp.json()
                    error = error_data.get("error")
                    if error == "authorization_pending":
                        pass # Keep polling
                    elif error == "slow_down":
                        interval += 5
                    elif error == "expired_token":
                        print("\nThe device code has expired. Please try again.")
                        return False
                    elif error == "access_denied":
                        print("\nAuthorization was denied.")
                        return False
                    else:
                        print(f"\nError polling for token: {error}")
                        return False
                else:
                    print(f"\nUnexpected status {token_resp.status_code} while polling.")
                    return False

                time.sleep(interval)

            print("\nTimed out waiting for authorization.")
            return False

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"\nError: The OAuth endpoint for '{provider}' is not available or has changed (404 Not Found).")
                print("If you are using a reverse-engineered or custom proxy endpoint, please update it in config.py.")
            else:
                print(f"\nHTTP error during OAuth flow: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error during OAuth flow: {e}")
            return False

    def perform_authorization_code_flow(self, provider: str, config: Dict[str, Any]) -> bool:
        """Perform the OAuth2 Authorization Code Flow using a local redirect server."""
        client_id = config.get("client_id")
        client_secret = config.get("client_secret", "")
        auth_url_base = config.get("authorization_url")
        redirect_uri = config.get("redirect_uri", "http://localhost:5000/callback")

        state = os.urandom(16).hex()

        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state
        }

        if "scope" in config:
            params["scope"] = config["scope"]

        auth_url = f"{auth_url_base}?{urllib.parse.urlencode(params)}"

        print("\n" + "=" * 50)
        print("Action Required: OAuth Authorization (Browser Flow)")
        print("=" * 50)
        print(f"1. A browser window will now open automatically to: \n   {auth_url}")
        print(f"   (If it doesn't open, please click the link above.)")
        print("2. Please authorize the application in your browser.")
        print("\nWaiting for authorization callback on localhost:5000...")

        try:
            webbrowser.open(auth_url)
        except Exception as e:
            logging.debug(f"Could not open browser automatically: {e}")

        server = HTTPServer(('localhost', 5000), OAuthCallbackHandler)
        server.oauth_code = None
        server.oauth_state = None
        server.oauth_error = False

        # Keep handling requests until we get a callback with a code or an error
        while not server.oauth_code and not server.oauth_error:
            server.handle_request()

        if not server.oauth_code:
            print("\nFailed to receive authorization code.")
            return False

        if server.oauth_state != state:
            print("\nSecurity warning: OAuth state mismatch!")
            return False

        print("\nAuthorization code received. Exchanging for token...")

        token_data = {
            "grant_type": "authorization_code",
            "code": server.oauth_code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
        }
        if client_secret:
            token_data["client_secret"] = client_secret

        try:
            token_resp = requests.post(config["token_url"], data=token_data, timeout=10)
            token_resp.raise_for_status()
            token_json = token_resp.json()

            access_token = token_json.get("access_token")
            if not access_token:
                print("\nFailed to retrieve access token from response.")
                return False

            refresh_token = token_json.get("refresh_token")
            expires_in = token_json.get("expires_in", 3600)

            tokens = self._load_tokens()
            tokens[provider] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": time.time() + expires_in
            }
            self._save_tokens(tokens)
            print("\nSuccessfully logged in!")
            return True

        except requests.exceptions.RequestException as e:
            print(f"\nFailed to exchange token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return False

    def perform_flow(self, provider: str) -> bool:
        """Perform OAuth for a selected provider."""
        if provider not in OAUTH_PROVIDERS:
            print(f"Provider {provider} is not supported for OAuth.")
            return False

        config = OAUTH_PROVIDERS[provider]
        print(f"\nInitiating login for {provider}...")

        if "device_authorization_url" in config:
            return self.perform_device_code_flow(provider, config)
        elif "authorization_url" in config:
            return self.perform_authorization_code_flow(provider, config)
        else:
            print(f"Provider {provider} has no supported OAuth flows configured.")
            return False

    def display_config_menu(self) -> None:
        """Display an interactive configuration menu for OAuth logins."""
        print("\n" + "=" * 50)
        print("renAI OAuth Configuration")
        print("=" * 50)
        print("Select a provider to log in with your subscription:")

        providers = list(OAUTH_PROVIDERS.keys())
        for i, provider in enumerate(providers, 1):
            status = "[Logged In]" if self.get_token(provider) else "[Not Logged In]"
            print(f"{i}. {provider.capitalize()} {status}")

        print(f"{len(providers) + 1}. Exit")

        while True:
            try:
                choice = input("\nEnter your choice: ")
                idx = int(choice) - 1

                if idx == len(providers):
                    break

                if 0 <= idx < len(providers):
                    selected = providers[idx]
                    self.perform_flow(selected)
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
