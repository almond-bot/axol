#include <string.h>
#include <stdio.h>
#include "tusb.h"
#include "gs_usb.h"
#include "pico/unique_id.h"

// USB device descriptor
tusb_desc_device_t const desc_device = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = 0x0200,  // USB 2.0
    .bDeviceClass       = 0xFF,    // Vendor class
    .bDeviceSubClass    = 0xFF,
    .bDeviceProtocol    = 0xFF,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor           = GS_USB_VID,
    .idProduct          = GS_USB_PID,
    .bcdDevice          = AXOL_HUB_FW_VERSION,  // firmware version (CMakeLists.txt)
    .iManufacturer      = 0x01,
    .iProduct           = 0x02,
    .iSerialNumber      = 0x03,
    .bNumConfigurations = 0x01
};

// Invoked when received GET DEVICE DESCRIPTOR request
uint8_t const* tud_descriptor_device_cb(void) {
    return (uint8_t const*)&desc_device;
}

// Serial number string populated from unique flash ID at runtime
static char serial_str[17];  // 16 hex chars + null terminator

// String descriptors
char const* string_desc_arr[] = {
    (const char[]){0x09, 0x04},  // 0: Supported languages (English)
    "Almond",                    // 1: Manufacturer
    "Axol CAN Bus",               // 2: Product
    serial_str,                  // 3: Serial number (set in usb_descriptors_init)
};

void usb_descriptors_init(void) {
    pico_unique_board_id_t board_id;
    pico_get_unique_board_id(&board_id);
    snprintf(serial_str, sizeof(serial_str),
             "%02X%02X%02X%02X%02X%02X%02X%02X",
             board_id.id[0], board_id.id[1], board_id.id[2], board_id.id[3],
             board_id.id[4], board_id.id[5], board_id.id[6], board_id.id[7]);
}

static uint16_t _desc_str[32];

// Invoked when received GET STRING DESCRIPTOR request
uint16_t const* tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;

    uint8_t chr_count;

    if (index == 0) {
        memcpy(&_desc_str[1], string_desc_arr[0], 2);
        chr_count = 1;
    } else {
        if (!(index < sizeof(string_desc_arr) / sizeof(string_desc_arr[0])))
            return NULL;

        const char* str = string_desc_arr[index];
        chr_count = strlen(str);
        if (chr_count > 31)
            chr_count = 31;

        for (uint8_t i = 0; i < chr_count; i++) {
            _desc_str[1 + i] = str[i];
        }
    }

    _desc_str[0] = (TUSB_DESC_STRING << 8) | (2 * chr_count + 2);
    return _desc_str;
}

enum {
    ITF_NUM_VENDOR,
    ITF_NUM_TOTAL
};

#define EPNUM_VENDOR_OUT 0x01
#define EPNUM_VENDOR_IN  0x81

#define CONFIG_TOTAL_LEN (TUD_CONFIG_DESC_LEN + TUD_VENDOR_DESC_LEN)

uint8_t const desc_configuration[] = {
    TUD_CONFIG_DESCRIPTOR(1, ITF_NUM_TOTAL, 0, CONFIG_TOTAL_LEN,
                          TUSB_DESC_CONFIG_ATT_REMOTE_WAKEUP, 100),

    TUD_VENDOR_DESCRIPTOR(ITF_NUM_VENDOR, 0, EPNUM_VENDOR_OUT, EPNUM_VENDOR_IN,
                          CFG_TUD_VENDOR_EPSIZE),
};

// Invoked when received GET CONFIGURATION DESCRIPTOR
uint8_t const* tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_configuration;
}
