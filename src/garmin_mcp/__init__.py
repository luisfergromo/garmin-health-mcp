"""
Modular MCP Server for Garmin Connect Data

Compatible with Gemini (Antigravity IDE), Claude Desktop, and other MCP clients.
"""

import os
import sys
import base64

from dotenv import load_dotenv
import requests
from mcp.server.fastmcp import FastMCP

# Load .env file before reading any environment variables
load_dotenv()

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

# Import all modules
from garmin_mcp import token_utils
from garmin_mcp import activity_management
from garmin_mcp import health_wellness
from garmin_mcp import user_profile
from garmin_mcp import devices
from garmin_mcp import gear_management
from garmin_mcp import weight_management
from garmin_mcp import training
from garmin_mcp import workouts


def is_interactive_terminal() -> bool:
    """Detect if running in interactive terminal vs MCP subprocess.

    Returns:
        bool: True if running in an interactive terminal, False otherwise
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def get_mfa() -> str:
    """Get MFA code from user input.

    Raises:
        RuntimeError: If running in non-interactive environment
    """
    if not is_interactive_terminal():
        print(
            "\nERROR: MFA code required but no interactive terminal available.\n"
            "Please run 'garmin-mcp-auth' in your terminal first.\n"
            "See: https://github.com/Taxuspt/garmin_mcp#mfa-setup\n",
            file=sys.stderr,
        )
        raise RuntimeError("MFA required but non-interactive environment")

    print(
        "\nGarmin Connect MFA required. Please check your email/phone for the code.",
        file=sys.stderr,
    )
    return input("Enter MFA code: ")


# Get credentials from environment
email = os.environ.get("GARMIN_EMAIL")
email_file = os.environ.get("GARMIN_EMAIL_FILE")
if email and email_file:
    raise ValueError(
        "Must only provide one of GARMIN_EMAIL and GARMIN_EMAIL_FILE, got both"
    )
elif email_file:
    with open(email_file, "r") as _ef:
        email = _ef.read().rstrip()

password = os.environ.get("GARMIN_PASSWORD")
password_file = os.environ.get("GARMIN_PASSWORD_FILE")
if password and password_file:
    raise ValueError(
        "Must only provide one of GARMIN_PASSWORD and GARMIN_PASSWORD_FILE, got both"
    )
elif password_file:
    with open(password_file, "r") as _pf:
        password = _pf.read().rstrip()

tokenstore = token_utils.get_token_path()
tokenstore_base64 = token_utils.get_token_base64_path()
is_cn = os.getenv("GARMIN_IS_CN", "false").lower() in ("true", "1", "yes")


# --- Tool filtering ---------------------------------------------------------
def _parse_tool_set(value):
    if not value:
        return set()
    return {name.strip().lower() for name in value.split(",") if name.strip()}


enabled_tools = _parse_tool_set(os.getenv("GARMIN_ENABLED_TOOLS"))
disabled_tools = _parse_tool_set(os.getenv("GARMIN_DISABLED_TOOLS"))


_VALID_TRANSPORTS = ("stdio", "streamable-http", "sse")


class _GarminProxy:
    """Wraps the Garmin client to translate known runtime exceptions into clear messages.

    Without this, token expiry or rate-limiting during a tool call surfaces raw
    library tracebacks to the MCP client. The proxy intercepts each attribute
    access and, if the result is callable, wraps the call so that known Garmin
    exceptions become user-friendly strings rather than server errors.
    """

    _MESSAGES = {
        GarminConnectAuthenticationError: (
            "Garmin authentication expired. "
            "Re-run 'garmin-mcp-auth' to refresh your tokens and restart the server."
        ),
        GarminConnectTooManyRequestsError: (
            "Garmin rate limit hit. Wait a few minutes before retrying."
        ),
        GarminConnectConnectionError: (
            "Garmin Connect is unreachable. Check your network connection or try again later."
        ),
    }

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            try:
                return attr(*args, **kwargs)
            except tuple(self._MESSAGES) as exc:
                for exc_type, msg in self._MESSAGES.items():
                    if isinstance(exc, exc_type):
                        error_details = str(exc)
                        full_msg = (
                            f"{msg} (Details: {error_details})"
                            if error_details
                            else msg
                        )
                        raise type(exc)(full_msg) from None
                raise

        return _call


def _parse_transport_config() -> tuple[str, str, int]:
    """Read and validate HTTP transport env vars. Raises ValueError on bad input."""
    transport = os.getenv("GARMIN_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(
            f"GARMIN_MCP_TRANSPORT must be one of {_VALID_TRANSPORTS}, got '{transport}'"
        )
    host = os.getenv("GARMIN_MCP_HOST", "127.0.0.1").strip()
    try:
        port = int(os.getenv("GARMIN_MCP_PORT", "8000"))
    except ValueError:
        raise ValueError("GARMIN_MCP_PORT must be an integer")
    return transport, host, port


def _init_garmin_client() -> _GarminProxy:
    """Initialize and authenticate the Garmin Connect client."""
    
    # Try base64 token string from environment variable (Best for Render/Cloud)
    env_b64 = os.getenv("GARMIN_TOKENS_BASE64") or os.getenv("GARMINTOKENS_BASE64")
    if env_b64 and len(env_b64.strip()) > 50:
        try:
            # Init without credentials to avoid fallback to login and 429 IP block
            garmin = Garmin(email="", password="", is_cn=is_cn, prompt_mfa=get_mfa)
            token_json_str = base64.b64decode(env_b64.strip()).decode("utf-8")
            expanded_store = token_utils.resolve_token_path(tokenstore)
            os.makedirs(expanded_store, exist_ok=True)
            token_file = os.path.join(expanded_store, "garmin_tokens.json")
            with open(token_file, "w") as f:
                f.write(token_json_str)
            garmin.login(expanded_store)
            print("Authenticated via GARMIN_TOKENS_BASE64 environment variable.", file=sys.stderr)
            return _GarminProxy(garmin)
        except Exception as e:
            print(f"GARMIN_TOKENS_BASE64 auth failed ({e}), falling back...", file=sys.stderr)

    # Try token-based auth from local disk
    if token_utils.token_exists(tokenstore):
        try:
            # Init without credentials to avoid automatic fallback
            garmin = Garmin(email="", password="", is_cn=is_cn, prompt_mfa=get_mfa)
            garmin.login(tokenstore)
            print("Authenticated via saved tokens.", file=sys.stderr)
            return _GarminProxy(garmin)
        except Exception as e:
            print(
                f"Token auth failed ({e}), falling back...",
                file=sys.stderr,
            )

    # Try base64 token file
    expanded_b64 = token_utils.resolve_token_path(tokenstore_base64)
    if os.path.exists(expanded_b64):
        try:
            with open(expanded_b64, "r") as f:
                token_b64 = f.read().strip()
            token_data = base64.b64decode(token_b64).decode()
            # Init without credentials to avoid automatic fallback
            garmin = Garmin(email="", password="", is_cn=is_cn, prompt_mfa=get_mfa)
            garmin.login(token_data)
            print("Authenticated via base64 tokens file.", file=sys.stderr)
            return _GarminProxy(garmin)
        except Exception as e:
            print(
                f"Base64 token auth failed ({e}), falling back to credentials...",
                file=sys.stderr,
            )

    # Fall back to email/password login only if tokens fail or are missing
    if email and password:
        try:
            garmin = Garmin(email=email, password=password, is_cn=is_cn, prompt_mfa=get_mfa)
            garmin.login()
            # Save tokens for next time
            expanded_store = token_utils.resolve_token_path(tokenstore)
            garmin.client.dump(expanded_store)
            token_utils.secure_token_dir(tokenstore)
            print("Authenticated via credentials. Tokens saved.", file=sys.stderr)
            return _GarminProxy(garmin)
        except Exception as e:
            print(f"Credential auth failed: {e}", file=sys.stderr)
            raise

    raise RuntimeError(
        "No authentication method available. "
        "Run 'garmin-mcp-auth' first, or set GARMIN_EMAIL and GARMIN_PASSWORD."
    )


# --- Module registry ---------------------------------------------------------
_ALL_MODULES = [
    activity_management,
    health_wellness,
    training,
    devices,
    gear_management,
    weight_management,
    workouts,
    user_profile,
]


# Create MCP server instance
mcp = FastMCP("garmin")


def _configure_and_register(client):
    """Configure all modules with the client and register their tools."""
    for module in _ALL_MODULES:
        module.configure(client)
        module.register_tools(mcp)


def main():
    """Entry point for the garmin-mcp server."""
    transport, host, port = _parse_transport_config()

    # Initialize Garmin client
    try:
        client = _init_garmin_client()
    except Exception as e:
        print(f"\nFATAL: Could not authenticate with Garmin Connect: {e}", file=sys.stderr)
        print(
            "\nTo fix this, run: garmin-mcp-auth\n"
            "Or set environment variables: GARMIN_EMAIL and GARMIN_PASSWORD\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure modules and register tools
    _configure_and_register(client)

    # Apply tool filtering
    if enabled_tools or disabled_tools:
        registered = list(mcp._tool_manager._tools.keys()) if hasattr(mcp, '_tool_manager') else []
        for tool_name in registered:
            lower_name = tool_name.lower()
            if enabled_tools and lower_name not in enabled_tools:
                mcp._tool_manager._tools.pop(tool_name, None)
                print(f"Filtered out tool: {tool_name}", file=sys.stderr)
            elif disabled_tools and lower_name in disabled_tools:
                mcp._tool_manager._tools.pop(tool_name, None)
                print(f"Filtered out tool: {tool_name}", file=sys.stderr)

    # Start the server
    print(f"Starting Garmin MCP server (transport={transport})...", file=sys.stderr)

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=transport, host=host, port=port)
