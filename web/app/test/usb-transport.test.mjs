import assert from "node:assert/strict"
import test from "node:test"

import { hostCertAuthorizeVisible, usbCertOrigin } from "../src/lib/usb-transport.ts"

test("the USB certificate is always authorized on localhost, never the LAN address", () => {
  assert.equal(usbCertOrigin(8000), "https://localhost:8000")
  assert.equal(usbCertOrigin(8001), "https://localhost:8001")
})

test("the host certificate button stays available in USB mode", () => {
  assert.equal(hostCertAuthorizeVisible("axol-host.local"), true)
})

test("the host certificate button needs a host to authorize", () => {
  assert.equal(hostCertAuthorizeVisible(""), false)
  assert.equal(hostCertAuthorizeVisible("   "), false)
})
