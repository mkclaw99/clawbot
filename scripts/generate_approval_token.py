"""
Generates a cryptographic approval token for the human gate.
The raw secret is shown ONCE — save it securely.
The SHA-256 hash is what goes in .env as APPROVAL_TOKEN_HASH.

Usage:
    python scripts/generate_approval_token.py
"""
import hashlib
import secrets
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


if __name__ == "__main__":
    token  = secrets.token_hex(32)
    digest = hashlib.sha256(token.encode()).hexdigest()

    print("=" * 60)
    print("APPROVAL TOKEN GENERATED")
    print("=" * 60)
    print()
    print(f"  Raw secret  (SAVE THIS — shown only once):")
    print(f"  {token}")
    print()
    print(f"  Hash for .env:")
    print(f"  APPROVAL_TOKEN_HASH={digest}")
    print()
    print("  Add the hash line to your .env file.")
    print("  Store the raw secret somewhere safe (password manager).")
    print("  The raw secret is required to:")
    print("    • Reset a Level 3 circuit breaker")
    print("    • Approve individual large trades")
    print("    • Enable live trading (future phase)")
    print("=" * 60)
