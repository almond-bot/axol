from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from almond_axol.utils.network import local_ip


class LocalIpTest(unittest.TestCase):
    def test_isolated_host_falls_back_without_raising(self) -> None:
        sock = MagicMock()
        sock.__enter__.return_value = sock
        sock.connect.side_effect = OSError("network unreachable")
        with (
            patch("almond_axol.utils.network.socket.socket", return_value=sock),
            patch(
                "almond_axol.utils.network.socket.gethostbyname",
                side_effect=OSError("no resolver"),
            ),
        ):
            self.assertEqual(local_ip(), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
