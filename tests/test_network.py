from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from almond_axol.cli.serve import _self_hosted_origin_hosts
from almond_axol.utils.network import local_interface_ips, local_ip


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

    def test_interface_inventory_includes_secondary_ipv4_and_ipv6(self) -> None:
        inventory = [
            {
                "addr_info": [
                    {"family": "inet", "local": "192.168.1.10"},
                    {"family": "inet", "local": "192.168.50.10"},
                    {"family": "inet6", "local": "fd12:3456::10"},
                    {"family": "inet6", "local": "fe80::1234%eth0"},
                    {"family": "inet", "local": "0.0.0.0"},
                    {"family": "inet", "local": "not-an-ip"},
                ]
            }
        ]
        with (
            patch(
                "almond_axol.utils.network._trusted_ip_command",
                return_value="/usr/sbin/ip",
            ),
            patch(
                "almond_axol.utils.network.subprocess.run",
                return_value=Mock(returncode=0, stdout=json.dumps(inventory)),
            ) as command,
            patch("almond_axol.utils.network.socket.getaddrinfo") as resolve,
        ):
            self.assertEqual(
                local_interface_ips(),
                frozenset({"192.168.1.10", "192.168.50.10", "fd12:3456::10"}),
            )
        resolve.assert_not_called()
        self.assertEqual(command.call_args.args[0][0], "/usr/sbin/ip")

    def test_interface_inventory_falls_back_to_hostname_resolution(self) -> None:
        with (
            patch("almond_axol.utils.network._trusted_ip_command", return_value=None),
            patch("almond_axol.utils.network.subprocess.run") as command,
            patch(
                "almond_axol.utils.network.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("172.20.0.4", 0))],
            ),
            patch("almond_axol.utils.network._locally_bindable_ip", return_value=True),
        ):
            self.assertEqual(local_interface_ips(), frozenset({"172.20.0.4"}))
        command.assert_not_called()

    def test_interface_inventory_rejects_oversized_command_output(self) -> None:
        with (
            patch(
                "almond_axol.utils.network._trusted_ip_command",
                return_value="/usr/bin/ip",
            ),
            patch(
                "almond_axol.utils.network.subprocess.run",
                return_value=Mock(returncode=0, stdout="[]oversized"),
            ) as command,
            patch("almond_axol.utils.network._IP_INVENTORY_MAX_CHARS", 2),
            patch("almond_axol.utils.network.json.loads") as decode,
        ):
            self.assertEqual(local_interface_ips(), frozenset())

        decode.assert_not_called()
        self.assertEqual(command.call_args.args[0][0], "/usr/bin/ip")

    def test_hostname_fallback_rejects_an_address_not_bound_locally(self) -> None:
        with (
            patch("almond_axol.utils.network._trusted_ip_command", return_value=None),
            patch(
                "almond_axol.utils.network.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("203.0.113.9", 0))],
            ),
            patch("almond_axol.utils.network._locally_bindable_ip", return_value=False),
        ):
            self.assertEqual(local_interface_ips(), frozenset())

    def test_wildcard_serve_registers_every_local_interface_origin(self) -> None:
        with (
            patch(
                "almond_axol.cli.serve.local_interface_ips",
                return_value=frozenset({"192.168.1.10", "192.168.50.10", "fd12::10"}),
            ),
            patch("almond_axol.cli.serve.socket.gethostname", return_value="axol"),
            patch("almond_axol.cli.serve.socket.getfqdn", return_value="axol.local"),
        ):
            hosts = _self_hosted_origin_hosts(
                bind_host="0.0.0.0", lan_ip="192.168.1.10"
            )

        self.assertTrue(
            {"192.168.1.10", "192.168.50.10", "fd12::10", "axol", "axol.local"} <= hosts
        )

    def test_explicit_bind_does_not_trust_unrelated_interface_addresses(self) -> None:
        with (
            patch("almond_axol.cli.serve.local_interface_ips") as inventory,
            patch("almond_axol.cli.serve.socket.gethostname", return_value="axol"),
            patch("almond_axol.cli.serve.socket.getfqdn", return_value="axol.local"),
        ):
            hosts = _self_hosted_origin_hosts(
                bind_host="192.168.50.10", lan_ip="192.168.50.10"
            )

        inventory.assert_not_called()
        self.assertIn("192.168.50.10", hosts)


if __name__ == "__main__":
    unittest.main()
