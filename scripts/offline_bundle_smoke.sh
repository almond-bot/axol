#!/usr/bin/env bash
# Install an offline update bundle with no network access and check that it works.
#
# The release workflow runs this inside `docker run --network none` (see
# .github/workflows/publish.yml), in an image prepared like a robot: uv on
# PATH, Python 3.13 under /opt/axol/uv/python, and the apt packages a
# provisioned host already has. Each check runs the same code path a robot
# runs, so a bundle missing anything fails the release, not a customer.
#
#   scripts/offline_bundle_smoke.sh /bundle/axol-offline-v0.2.47-linux-aarch64
set -euo pipefail

BUNDLE="${1:?usage: offline_bundle_smoke.sh BUNDLE_DIR}"
BUNDLE="$(cd "${BUNDLE}" && pwd)"
export UV_TOOL_DIR=/opt/axol/uv/tools
export UV_PYTHON_INSTALL_DIR=/opt/axol/uv/python
export UV_TOOL_BIN_DIR=/usr/local/bin
export UV_OFFLINE=1
TOOL_PY="${UV_TOOL_DIR}/almond-axol/bin/python"

step() { printf '\n== %s\n' "$*"; }
value() { sed -n "s/^$1=//p" "${BUNDLE}/bundle.env"; }

step "Network is off"
if curl -fsS --max-time 5 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
    echo "pypi.org is reachable; run this with --network none" >&2
    exit 1
fi

step "Checksums"
(cd "${BUNDLE}" && sha256sum --quiet --strict -c SHA256SUMS)

VERSION="$(value AXOL_BUNDLE_VERSION)"
EXTRAS="$(value AXOL_BUNDLE_EXTRAS)"
PYTHON="$(value AXOL_BUNDLE_PYTHON)"

step "uv tool install from the bundle (as web/app/public/install does)"
uv tool install --python "${PYTHON}" --force \
    --offline --no-index --find-links "${BUNDLE}/wheels" \
    "almond-axol[${EXTRAS}]==${VERSION}"

step "Counts as a release install of v${VERSION}"
"${TOOL_PY}" - "${VERSION}" <<'EOF'
import sys
from almond_axol.serve import update
assert update.installed_version() == sys.argv[1], update.installed_version()
assert update.installed_from_index(), "find-links install wrote a direct_url.json"
EOF

step "Bundle is signed with a trusted key and valid for this host"
"${TOOL_PY}" - "${BUNDLE}" <<'EOF'
import sys
from almond_axol import offline_bundle
bundle = offline_bundle.load_signed(sys.argv[1])
bundle.check_host()
bundle.verify()
EOF

step "axol-rt installs from the bundle"
AXOL_OFFLINE_BUNDLE="${BUNDLE}" axol rt.install
test -x /usr/local/bin/axol-rt
grep -qx "v${VERSION}" /root/.almond/axol-rt-src/.installed-ref-* 2>/dev/null \
    || grep -qx "v${VERSION}" /root/.almond/axol-rt-src/.installed-ref

step "PyGObject builds from the bundle"
AXOL_OFFLINE_BUNDLE="${BUNDLE}" "${TOOL_PY}" - <<'EOF'
import logging
logging.basicConfig(level=logging.INFO)
from almond_axol.cli.gst import install
assert install._pip_install_pygobject(), "PyGObject did not install offline"
EOF
"${TOOL_PY}" -c "import gi; gi.require_version('GLib', '2.0'); from gi.repository import GLib"

step "Pinned libsurvive and zed-gstreamer sources check out from the bundle"
AXOL_OFFLINE_BUNDLE="${BUNDLE}" "${TOOL_PY}" - <<'EOF'
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
from almond_axol.cli import tracker_install
from almond_axol.cli.gst import build_zed
assert tracker_install._sync_source(Path("/tmp/libsurvive")), "libsurvive"
assert build_zed._sync_source(Path("/tmp/zed-gstreamer")), "zed-gstreamer"
assert build_zed._apply_patches(Path("/tmp/zed-gstreamer")), "zed-gstreamer patches"
EOF

step "axol serve hosts the control panel and VR app"
axol serve --no-tls --host 127.0.0.1 --port 8001 >/tmp/serve.log 2>&1 &
SERVE_PID=$!
trap 'kill ${SERVE_PID} 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do
    curl -fsS -o /dev/null http://127.0.0.1:8001/control && break
    kill -0 "${SERVE_PID}" 2>/dev/null || { cat /tmp/serve.log; exit 1; }
    sleep 1
done
for path in /control /vr; do
    curl -fsS http://127.0.0.1:8001${path} | grep -qi "<!doctype html" \
        || { echo "${path} did not serve the web UI" >&2; cat /tmp/serve.log; exit 1; }
done

step "Offline bundle v${VERSION} works with no network"
