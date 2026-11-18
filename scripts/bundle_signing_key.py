"""Create the offline bundle signing key and store it in GitHub.

The release workflow signs every offline bundle's manifest.json with an
ed25519 key from the ``offline-bundle`` GitHub environment, and robots only
install uploads signed by a key in ``almond_axol.offline_bundle.TRUSTED_KEYS``.
This script makes a new key pair and hands the private key straight to
``gh secret set`` on stdin: it is never written to disk or printed. It prints
the public key line to add to TRUSTED_KEYS.

    uv run --with cryptography python scripts/bundle_signing_key.py

Needs ``gh`` authenticated as a repository admin. It also creates the
``offline-bundle`` environment, limited to release tags (``v*``), so only the
release workflow can read the secret. Rotating: add the new public key to
TRUSTED_KEYS and release that first, then run this again to replace the secret.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from almond_axol import offline_bundle as ob  # noqa: E402

ENVIRONMENT = "offline-bundle"
SECRET = "AXOL_BUNDLE_SIGNING_KEY"


def gh(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="almond-bot/axol")
    args = parser.parse_args()

    # The environment, deployable only from release tags.
    gh(
        "api",
        "--method",
        "PUT",
        f"repos/{args.repo}/environments/{ENVIRONMENT}",
        "--input",
        "-",
        stdin=json.dumps(
            {
                "deployment_branch_policy": {
                    "protected_branches": False,
                    "custom_branch_policies": True,
                }
            }
        ),
    )
    try:
        gh(
            "api",
            "--method",
            "POST",
            f"repos/{args.repo}/environments/{ENVIRONMENT}/deployment-branch-policies",
            "-f",
            "name=v*",
            "-f",
            "type=tag",
        )
    except subprocess.CalledProcessError as exc:
        if "already exists" not in (exc.stderr or ""):
            raise

    private = Ed25519PrivateKey.generate()
    seed = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    public_b64 = base64.b64encode(public).decode()
    gh(
        "secret",
        "set",
        SECRET,
        "--repo",
        args.repo,
        "--env",
        ENVIRONMENT,
        stdin=base64.b64encode(seed).decode(),
    )
    print(f"Stored {SECRET} in the {ENVIRONMENT} environment of {args.repo}.")
    print("Add this to TRUSTED_KEYS in almond_axol/offline_bundle.py:")
    print(f'    "{ob.key_id(public_b64)}": "{public_b64}",')


if __name__ == "__main__":
    main()
