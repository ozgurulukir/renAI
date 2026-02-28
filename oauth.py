import json
import logging
import os
import time
import requests
from typing import Optional, Dict, Any
from pathlib import Path
from config import OAUTH_PROVIDERS, AUTH_DIR
from utils import ensure_secure_dir

TOKEN_FILE = AUTH_DIR / "tokens.json"

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

    def perform_device_code_flow(self, provider: str) -> bool:
        """Perform the OAuth2 Device Code Flow for a selected provider."""
        if provider not in OAUTH_PROVIDERS:
            print(f"Provider {provider} is not supported for OAuth.")
            return False

        config = OAUTH_PROVIDERS[provider]
        client_id = config.get("client_id")

        print(f"\nInitiating login for {provider}...")

        try:
            # 1. Request device code
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
            print("Action Required: OAuth Authorization")
            print("=" * 50)
            print(f"1. Open this URL in your browser:\n   {verification_uri}")
            print(f"2. Enter this code:\n   {user_code}")
            print("\nWaiting for authorization...")

            # 2. Poll for token
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

        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error during OAuth flow: {e}")
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
                    self.perform_device_code_flow(selected)
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
