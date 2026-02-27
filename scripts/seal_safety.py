"""
Run this script after intentionally modifying safety_constants.py.
It recomputes the SHA-256 hash and saves it to config/safety.hash.

Usage:
    python scripts/seal_safety.py

This is a DELIBERATE HUMAN action — the bot cannot run this.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.safety import seal_safety_constants

if __name__ == "__main__":
    print("⚠️  You are resealing the safety constants.")
    print("    This should only be done after a deliberate, reviewed change.")
    print()
    confirm = input("Type 'SEAL' to confirm: ").strip()
    if confirm != "SEAL":
        print("Aborted.")
        sys.exit(1)

    digest = seal_safety_constants()
    print(f"\n✅ Safety constants sealed.")
    print(f"   Hash: {digest}")
    print(f"   Saved to: config/safety.hash")
