#ifndef GS_USB_H
#define GS_USB_H

#include <stdint.h>
#include <stdbool.h>

// gs_usb protocol definitions matching Linux kernel driver
// Based on drivers/net/can/usb/gs_usb.c

// Vendor ID and Product ID for gs_usb auto-binding
#define GS_USB_VID 0x1D50  // Openmoko
#define GS_USB_PID 0x606F  // candleLight

// USB control request types (BREQ = bRequest)
enum gs_usb_breq {
    GS_USB_BREQ_HOST_FORMAT = 0,
    GS_USB_BREQ_BITTIMING,
    GS_USB_BREQ_MODE,
    GS_USB_BREQ_BERR,
    GS_USB_BREQ_BT_CONST,
    GS_USB_BREQ_DEVICE_CONFIG,
    GS_USB_BREQ_TIMESTAMP,
    GS_USB_BREQ_IDENTIFY,
    GS_USB_BREQ_GET_USER_ID,
    GS_USB_BREQ_SET_USER_ID,
    GS_USB_BREQ_DATA_BITTIMING,
    GS_USB_BREQ_BT_CONST_EXT,
};

// CAN mode flags
#define GS_CAN_MODE_RESET        0x00
#define GS_CAN_MODE_START        0x01
#define GS_CAN_MODE_LISTEN_ONLY  0x02
#define GS_CAN_MODE_LOOPBACK     0x04
#define GS_CAN_MODE_TRIPLE_SAMPLE 0x08
#define GS_CAN_MODE_ONE_SHOT     0x10

// CAN feature flags (for BT_CONST)
#define GS_CAN_FEATURE_LISTEN_ONLY   0x01
#define GS_CAN_FEATURE_LOOPBACK      0x02
#define GS_CAN_FEATURE_TRIPLE_SAMPLE 0x04
#define GS_CAN_FEATURE_ONE_SHOT      0x08
#define GS_CAN_FEATURE_HW_TIMESTAMP  0x10
#define GS_CAN_FEATURE_IDENTIFY      0x20
#define GS_CAN_FEATURE_USER_ID       0x40
#define GS_CAN_FEATURE_PAD_PKTS_TO_MAX_PKT_SIZE 0x80
#define GS_CAN_FEATURE_FD            0x100

// Host frame echo ID for RX frames
#define GS_HOST_FRAME_ECHO_ID_RX 0xFFFFFFFF

// Byte order (for HOST_FORMAT)
#define GS_USB_HOST_BYTE_ORDER_LITTLE_ENDIAN 0
#define GS_USB_HOST_BYTE_ORDER_BIG_ENDIAN   1

// Packed structures (little-endian as per USB spec)
#pragma pack(push, 1)

// Host configuration (HOST_FORMAT request)
struct gs_host_config {
    uint32_t byte_order;  // 0 = little-endian, 1 = big-endian
};

// Device configuration (DEVICE_CONFIG response)
struct gs_device_config {
    uint8_t reserved1;
    uint8_t reserved2;
    uint8_t reserved3;
    uint8_t icount;       // Channel count (kernel uses icount + 1, so 1 = 2 channels)
    uint32_t sw_version;   // Software version (LE)
    uint32_t hw_version;   // Hardware version (LE)
};

// Device mode (MODE request)
struct gs_device_mode {
    uint32_t mode;  // GS_CAN_MODE_* flags
    uint32_t flags; // Reserved, set to 0
};

// Device bit timing (BITTIMING request)
struct gs_device_bittiming {
    uint32_t prop_seg;    // Propagation segment
    uint32_t phase_seg1;  // Phase segment 1
    uint32_t phase_seg2;  // Phase segment 2
    uint32_t sjw;         // Synchronization jump width
    uint32_t brp;         // Bit rate prescaler
};

// Bit timing constants (BT_CONST response) - layout must match kernel gs_device_bt_const
struct gs_device_bt_const {
    uint32_t feature;     // GS_CAN_FEATURE_* flags (LE)
    uint32_t fclk_can;    // CAN clock frequency in Hz (LE)
    uint32_t tseg1_min;   // Time segment 1 minimum
    uint32_t tseg1_max;   // Time segment 1 maximum
    uint32_t tseg2_min;   // Time segment 2 minimum
    uint32_t tseg2_max;   // Time segment 2 maximum
    uint32_t sjw_max;     // Synchronization jump width maximum
    uint32_t brp_min;     // Bit rate prescaler minimum
    uint32_t brp_max;     // Bit rate prescaler maximum
    uint32_t brp_inc;     // Bit rate prescaler increment
};

// Host frame (bulk transfer format)
// 12-byte header + 8 bytes data = 20 bytes total for classic CAN
struct gs_host_frame {
    uint32_t echo_id;     // TX: host-provided, RX: GS_HOST_FRAME_ECHO_ID_RX
    uint32_t can_id;      // CAN ID (11-bit or 29-bit)
    uint8_t can_dlc;      // Data length code (0-8 for classic CAN)
    uint8_t channel;      // Channel index (0 or 1)
    uint8_t flags;        // CAN flags (extended ID, RTR, etc.)
    uint8_t reserved;     // Reserved
    uint8_t data[8];      // CAN data payload
};

#pragma pack(pop)

// Helper to compute bitrate from bittiming
// bitrate = fclk_can / ((1 + prop_seg + phase_seg1 + phase_seg2) * brp)
static inline uint32_t gs_compute_bitrate(const struct gs_device_bittiming *bt, uint32_t fclk_can) {
    uint32_t tq = 1 + bt->prop_seg + bt->phase_seg1 + bt->phase_seg2;
    return fclk_can / (tq * bt->brp);
}

#endif // GS_USB_H
