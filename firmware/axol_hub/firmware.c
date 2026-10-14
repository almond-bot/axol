#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "pico/bootrom.h"
#include "hardware/clocks.h"
#include "hardware/irq.h"
#include "hardware/structs/powman.h"
#include "hardware/structs/watchdog.h"
#include "can2040.h"
#include "tusb.h"
#include "device/dcd.h"  // DCD_EVENT_BUS_RESET for tud_event_hook_cb
#include "gs_usb.h"

// Defined in usb_descriptors.c
void usb_descriptors_init(void);

// GPIO definitions for CAN transceivers
// Channel 0 (left arm): PIO0
#define CAN0_GPIO_RX  17
#define CAN0_GPIO_TX  18

// Channel 1 (right arm): PIO1
#define CAN1_GPIO_RX  12
#define CAN1_GPIO_TX  13

// Status LED
#define STATUS_LED_GPIO  27

// Number of CAN channels
#define NUM_CHANNELS  2

// Firmware version, reported as the USB bcdDevice (readable unprivileged from
// sysfs) and as the gs_usb DEVICE_CONFIG sw_version.  Set in CMakeLists.txt;
// almond_axol.hub reads it to decide whether a hub needs reflashing.
#ifndef AXOL_HUB_FW_VERSION
#error AXOL_HUB_FW_VERSION must be defined (see CMakeLists.txt)
#endif

// Queue sizes (must be powers of 2)
#define RX_QUEUE_SIZE  128  // CAN RX → USB host (core1 writes, core0 reads)
#define TX_QUEUE_SIZE   64  // USB host → CAN TX (core0 writes, core1 reads)

//--------------------------------------------------------------------
// CAN channel state
// Core0 writes pending_start / pending_stop flags.
// Core1 owns all can2040 calls and polls these flags.
//--------------------------------------------------------------------

struct can_channel_state {
    struct can2040 cbus;
    volatile uint32_t bitrate;    // written by core0, read by core1
    volatile bool started;        // written by core1, read by core0
    volatile bool pending_start;  // written by core0, read by core1
    volatile bool pending_stop;   // written by core0, read by core1
    // Bumped by core0 on channel RESET / USB unmount.  Frames and echoes
    // are tagged with the epoch at enqueue time; consumers drop entries
    // whose epoch no longer matches, so traffic queued before a reset can
    // never replay onto the bus (or reach the host) after a restart.
    volatile uint32_t epoch;
    // TX stall recovery (all core1: tx_completed from the PIO IRQ, the rest
    // from the main loop).  CAN retransmits an unACKed frame forever at
    // ~100% bus duty; on this board a sustained storm on one channel drags
    // down the shared isolated 5V transceiver rail until the OTHER channel's
    // transceiver hits undervoltage lockout and goes mute.  So a frame that
    // cannot complete within TX_STALL_TIMEOUT_US is abandoned: the engine is
    // reinitialized and this channel's queued backlog is dropped.
    volatile uint32_t tx_completed;  // NOTIFY_TX count
    uint32_t tx_submitted;           // frames handed to can2040_transmit
    uint32_t stall_ref_completed;    // progress snapshot for stall timing
    uint32_t stall_ref_time;
    bool stall_armed;
    uint32_t drop_before;  // g_tx_queue positions before this are dropped
    bool drop_active;
    uint8_t rx_gpio;
    uint8_t tx_gpio;
};

// A healthy 1Mbps bus completes a frame in ~120us even under arbitration
// loss; 50ms of zero progress can only mean nothing is ACKing.
#define TX_STALL_TIMEOUT_US  50000

// A host frame that has not been handed to the CAN engine this long after it
// arrived is dropped instead.  Motor commands are setpoints: a stale one is
// worse than none.  This is the queue-side half of the rule that the hub
// never puts an old command on the wire (the engine-side half is the stall
// recovery above) — without it a frame parked behind a dead bus would go out
// the moment the motors are powered back on and snap the arm to it.
#define TX_MAX_AGE_US  50000

static struct can_channel_state g_channels[NUM_CHANNELS] = {
    {.rx_gpio = CAN0_GPIO_RX, .tx_gpio = CAN0_GPIO_TX, .bitrate = 0, .started = false},
    {.rx_gpio = CAN1_GPIO_RX, .tx_gpio = CAN1_GPIO_TX, .bitrate = 0, .started = false},
};

//--------------------------------------------------------------------
// RX queue: core1 IRQ callbacks write, core0 reads
//--------------------------------------------------------------------

struct rx_queue_entry {
    uint8_t channel;
    struct can2040_msg msg;
};

static struct {
    volatile uint32_t pull_pos;  // written by core0, read by core1 IRQ
    volatile uint32_t push_pos;  // written by core1 IRQ, read by core0
    struct rx_queue_entry entries[RX_QUEUE_SIZE];
} g_rx_queue;

//--------------------------------------------------------------------
// TX queue: core0 writes (USB RX), core1 reads and transmits on CAN
//--------------------------------------------------------------------

struct tx_queue_entry {
    struct gs_host_frame host_frame;
    uint32_t epoch;        // channel epoch at enqueue; stale entries are discarded
    uint32_t enqueued_us;  // time_us_32() at enqueue; see TX_MAX_AGE_US
};

static struct {
    volatile uint32_t push_pos;  // written by core0
    volatile uint32_t pull_pos;  // written by core1, read by core0 for backpressure
    struct tx_queue_entry entries[TX_QUEUE_SIZE];
} g_tx_queue;

//--------------------------------------------------------------------
// Echo queue: core0 pushes immediately on USB OUT (before CAN TX),
// and core0 drains to the USB host.  Immediate echoing keeps Linux
// tx_context slots cycling fast so Python's asyncio loop never blocks.
//--------------------------------------------------------------------

struct echo_queue_entry {
    struct gs_host_frame frame;
    uint32_t epoch;  // channel epoch at enqueue; stale entries are dropped
};

static struct {
    volatile uint32_t push_pos;  // written by core0 (process_one_bulk_out)
    volatile uint32_t pull_pos;  // written by core0 (process_echo_queue)
    struct echo_queue_entry entries[TX_QUEUE_SIZE];
} g_echo_queue;

// USB control request state
static uint8_t g_host_byte_order = GS_USB_HOST_BYTE_ORDER_LITTLE_ENDIAN;

//--------------------------------------------------------------------
// CAN2040 callbacks (run on core1 via PIO IRQ)
//--------------------------------------------------------------------

static void can2040_rx_cb_ch0(struct can2040 *cd, uint32_t notify, struct can2040_msg *msg) {
    (void)cd;
    if (notify == CAN2040_NOTIFY_TX) {
        g_channels[0].tx_completed++;
        return;
    }
    if (notify == CAN2040_NOTIFY_RX) {
        uint32_t push = g_rx_queue.push_pos;
        if ((push - g_rx_queue.pull_pos) >= RX_QUEUE_SIZE)
            return;  // queue full, drop
        g_rx_queue.entries[push % RX_QUEUE_SIZE].channel = 0;
        g_rx_queue.entries[push % RX_QUEUE_SIZE].msg = *msg;
        __dmb();
        g_rx_queue.push_pos = push + 1;
    }
}

static void can2040_rx_cb_ch1(struct can2040 *cd, uint32_t notify, struct can2040_msg *msg) {
    (void)cd;
    if (notify == CAN2040_NOTIFY_TX) {
        g_channels[1].tx_completed++;
        return;
    }
    if (notify == CAN2040_NOTIFY_RX) {
        uint32_t push = g_rx_queue.push_pos;
        if ((push - g_rx_queue.pull_pos) >= RX_QUEUE_SIZE)
            return;  // queue full, drop
        g_rx_queue.entries[push % RX_QUEUE_SIZE].channel = 1;
        g_rx_queue.entries[push % RX_QUEUE_SIZE].msg = *msg;
        __dmb();
        g_rx_queue.push_pos = push + 1;
    }
}


// PIO IRQ handlers — installed on core1
static void PIO0_IRQHandler(void) {
    can2040_pio_irq_handler(&g_channels[0].cbus);
}

static void PIO1_IRQHandler(void) {
    can2040_pio_irq_handler(&g_channels[1].cbus);
}

// Fully reinitialize a channel's can2040 engine (core1 only).
//
// can2040_start() after can2040_stop() reuses whatever internal state the
// engine had — tx_state and the internal transmit queue positions are only
// cleared by can2040_setup()'s memset.  If a stop lands mid-transmit (USB
// drop, host interface flap), a bare stop/start restart leaves tx_state
// latched: the engine then either never transmits again or reads back only
// its own frames, and the wedge persists until the chip loses power.
// Always rebuilding from setup makes every restart equivalent to a cold
// boot of the CAN engine.
//
// The channel's PIO NVIC interrupt is masked during the rebuild so the IRQ
// handler can never observe the struct mid-memset.
static void can_channel_reinit(uint8_t i) {
    struct can_channel_state *ch = &g_channels[i];
    uint32_t irq_num = (i == 0) ? PIO0_IRQ_0 : PIO1_IRQ_0;

    irq_set_enabled(irq_num, false);
    can2040_stop(&ch->cbus);
    can2040_setup(&ch->cbus, i);
    can2040_callback_config(&ch->cbus, (i == 0) ? can2040_rx_cb_ch0 : can2040_rx_cb_ch1);
    can2040_start(&ch->cbus, clock_get_hz(clk_sys), ch->bitrate, ch->rx_gpio, ch->tx_gpio);
    irq_set_enabled(irq_num, true);

    // The engine's internal queue was flushed; no NOTIFY_TX will arrive for
    // frames that were pending, so re-sync the progress counters.
    ch->tx_submitted = ch->tx_completed;
    ch->stall_armed = false;
}

// Stop a channel and discard everything it was about to send (core1 only).
//
// can2040_stop() alone halts the state machines but keeps the engine's
// internal transmit queue in RAM, and leaves the TX pin wherever the TX state
// machine last drove it — mid-frame that can be dominant, holding the bus.
// So the engine is wiped with can2040_setup() (the next start rebuilds it
// via can_channel_reinit() anyway) and the pin is parked recessive on SIO
// until can2040_start() hands it back to the PIO.
static void can_channel_halt(uint8_t i) {
    struct can_channel_state *ch = &g_channels[i];
    uint32_t irq_num = (i == 0) ? PIO0_IRQ_0 : PIO1_IRQ_0;

    irq_set_enabled(irq_num, false);
    can2040_stop(&ch->cbus);
    can2040_setup(&ch->cbus, i);
    can2040_callback_config(&ch->cbus, (i == 0) ? can2040_rx_cb_ch0 : can2040_rx_cb_ch1);
    gpio_init(ch->tx_gpio);
    gpio_put(ch->tx_gpio, 1);  // recessive
    gpio_set_dir(ch->tx_gpio, GPIO_OUT);
    irq_set_enabled(irq_num, true);  // engine irqs are masked (inte0 = 0) until start

    ch->tx_submitted = ch->tx_completed;
    ch->stall_armed = false;
}

// Detect a channel whose transmit path has made zero progress for
// TX_STALL_TIMEOUT_US despite pending frames (dead bus: nothing ACKs, the
// engine retransmits the head frame forever).  Recover by reinitializing
// the engine and dropping this channel's queued backlog, so the storm ends
// and the shared transceiver rail recovers for the other channel.
// Returns true when a recovery was performed.  Core1 only.
static bool check_tx_stall(uint8_t i) {
    struct can_channel_state *ch = &g_channels[i];
    uint32_t completed = ch->tx_completed;

    if (!ch->started || ch->tx_submitted == completed) {
        ch->stall_armed = false;
        return false;
    }
    uint32_t now = time_us_32();
    if (!ch->stall_armed || completed != ch->stall_ref_completed) {
        ch->stall_armed = true;
        ch->stall_ref_completed = completed;
        ch->stall_ref_time = now;
        return false;
    }
    if (now - ch->stall_ref_time < TX_STALL_TIMEOUT_US)
        return false;

    can_channel_reinit(i);
    ch->drop_before = g_tx_queue.push_pos;
    ch->drop_active = true;
    return true;
}

//--------------------------------------------------------------------
// Core1: CAN owner
// Registers IRQ handlers (installs on core1's NVIC), drains TX queue,
// and polls pending_start / pending_stop from core0.
//--------------------------------------------------------------------

static void core1_main(void) {
    // Register PIO IRQ handlers on core1's NVIC
    can2040_setup(&g_channels[0].cbus, 0);
    can2040_callback_config(&g_channels[0].cbus, can2040_rx_cb_ch0);
    irq_set_exclusive_handler(PIO0_IRQ_0, PIO0_IRQHandler);
    irq_set_priority(PIO0_IRQ_0, 1);
    irq_set_enabled(PIO0_IRQ_0, true);

    can2040_setup(&g_channels[1].cbus, 1);
    can2040_callback_config(&g_channels[1].cbus, can2040_rx_cb_ch1);
    irq_set_exclusive_handler(PIO1_IRQ_0, PIO1_IRQHandler);
    irq_set_priority(PIO1_IRQ_0, 1);
    irq_set_enabled(PIO1_IRQ_0, true);

    // Signal core0 that CAN setup is done
    multicore_fifo_push_blocking(1);

    while (true) {
        bool did_work = false;

        // Handle start/stop requests from core0.
        // Process stop before start so a RESET immediately followed by a
        // START (a restart) captured in the same pass leaves the channel
        // running rather than losing the START.
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            struct can_channel_state *ch = &g_channels[i];
            if (ch->pending_stop) {
                did_work = true;
                ch->pending_stop = false;
                if (ch->started) {
                    can_channel_halt(i);
                    __dmb();  // release: ensure the halt completes before started is visible to core0
                    ch->started = false;
                }
            }
            if (ch->pending_start) {
                did_work = true;
                ch->pending_start = false;
                if (!ch->started && ch->bitrate > 0) {
                    can_channel_reinit(i);
                    __dmb();  // release: ensure reinit completes before started is visible to core0
                    ch->started = true;
                }
            }
        }

        // TX stall recovery — must run even when there is no new work, since
        // a stuck engine generates PIO IRQs (retransmit attempts) that wake
        // the __wfe() below.
        for (uint8_t i = 0; i < NUM_CHANNELS; i++)
            if (check_tx_stall(i))
                did_work = true;

        // Drain TX queue → CAN bus
        // (Echoes were already sent immediately in process_one_bulk_out.)
        uint32_t push = g_tx_queue.push_pos;
        __dmb();  // acquire: ensure entry data written by core0 before push_pos++ is visible here
        if (g_tx_queue.pull_pos != push)
            did_work = true;
        while (g_tx_queue.pull_pos != push) {
            uint32_t pull = g_tx_queue.pull_pos;
            struct tx_queue_entry *entry = &g_tx_queue.entries[pull % TX_QUEUE_SIZE];
            uint8_t ch = entry->host_frame.channel;

            // Checks re-evaluated on every retry: a RESET (epoch), a stall
            // recovery (drop watermark) or simply waiting too long for a free
            // transmit slot (TX_MAX_AGE_US) invalidates it.
            bool valid = ch < NUM_CHANNELS;
            bool dropped = valid && g_channels[ch].drop_active &&
                           (int32_t)(pull - g_channels[ch].drop_before) < 0;
            bool expired = time_us_32() - entry->enqueued_us > TX_MAX_AGE_US;
            if (valid && g_channels[ch].started && !dropped && !expired &&
                entry->epoch == g_channels[ch].epoch) {
                struct can2040_msg msg;
                msg.id  = entry->host_frame.can_id;
                msg.dlc = entry->host_frame.can_dlc > 8 ? 8 : entry->host_frame.can_dlc;
                memcpy(msg.data, entry->host_frame.data, msg.dlc);

                if (can2040_transmit(&g_channels[ch].cbus, &msg) < 0) {
                    // If a start/stop was requested for ANY channel while we
                    // were waiting, break out so the outer loop can process it
                    // — a dead bus on this channel (no ACK, queue stays full)
                    // must not block requests for the other channel.
                    // pull_pos is unchanged, so this frame is retried after.
                    bool pending = false;
                    for (uint8_t i = 0; i < NUM_CHANNELS; i++)
                        if (g_channels[i].pending_stop || g_channels[i].pending_start)
                            pending = true;
                    if (pending)
                        break;
                    // Stall recovery must also run from here: with the engine
                    // queue full this loop never returns to the outer check.
                    bool recovered = false;
                    for (uint8_t i = 0; i < NUM_CHANNELS; i++)
                        recovered |= check_tx_stall(i);
                    if (recovered)
                        continue;  // re-evaluate: this frame may now be dropped
                    __wfe();  // sleep until NOTIFY_TX IRQ frees a slot
                    continue; // retry immediately
                }
                g_channels[ch].tx_submitted++;
            }
            // Invalid / stopped channel, stale epoch or expired: echo already
            // sent (or dropped as stale), nothing more to do.

            g_tx_queue.pull_pos = pull + 1;
        }

        // Nothing to do — sleep until core0 pushes a frame (sends __sev) or an IRQ fires.
        // WFE returns immediately if an event is already pending, so there is no race
        // with the __sev() in process_one_bulk_out().
        if (!did_work)
            __wfe();
    }
}

//--------------------------------------------------------------------
// USB Vendor control request handler (runs on core0)
//--------------------------------------------------------------------

static uint8_t g_control_buffer[64];

// Vendor-specific request (not part of the gs_usb protocol): reboot into the
// RP2350 USB bootloader so firmware can be flashed without physical BOOTSEL
// access.  Sent by a host tool via a control transfer; the Linux gs_usb
// driver never issues it.
#define AXOL_BREQ_BOOTLOADER  0xFF

// Vendor-specific request (IN): why the chip last reset, and how long it has
// been up.  The hub is bus-powered with RUN tied high and no watchdog, so an
// unexpected re-enumeration can only be a power event (POR/BOR) or a debug
// reset; this lets the host tell which, instead of inferring it.  The
// POWMAN_CHIP_RESET HAD_* bits describe only the most recent reset.  No
// counter can persist: a brown-out resets POWMAN scratch along with
// everything else.
#define AXOL_BREQ_RESET_INFO  0xFE
#define AXOL_RESET_INFO_VERSION  1

struct __attribute__((packed)) axol_reset_info {
    uint32_t version;          // AXOL_RESET_INFO_VERSION
    uint32_t chip_reset;       // POWMAN_CHIP_RESET captured at boot
    uint32_t watchdog_reason;  // WATCHDOG_REASON captured at boot
    uint32_t bod;              // POWMAN_BOD (brown-out detector config)
    uint64_t uptime_us;        // time since boot at the moment of the request
};

static uint32_t g_boot_chip_reset;
static uint32_t g_boot_watchdog_reason;
static struct axol_reset_info g_reset_info;

bool tud_vendor_control_xfer_cb(uint8_t rhport, uint8_t stage, tusb_control_request_t const *request) {
    (void)rhport;

    if (request->bmRequestType_bit.type != TUSB_REQ_TYPE_VENDOR)
        return false;

    uint8_t breq = request->bRequest;
    uint16_t wValue = request->wValue;
    uint16_t wLength = request->wLength;
    uint8_t channel = (uint8_t)wValue;

    if (stage == CONTROL_STAGE_SETUP) {
        switch (breq) {
            case GS_USB_BREQ_HOST_FORMAT:
                if (request->bmRequestType_bit.direction == TUSB_DIR_OUT && wLength <= sizeof(g_control_buffer))
                    return tud_control_xfer(rhport, request, g_control_buffer, wLength);
                return false;

            case GS_USB_BREQ_DEVICE_CONFIG:
                if (request->bmRequestType_bit.direction == TUSB_DIR_IN) {
                    struct gs_device_config cfg = {
                        .reserved1 = 0, .reserved2 = 0, .reserved3 = 0,
                        .icount = NUM_CHANNELS - 1,  // Linux gs_usb uses icount+1 as channel count
                        .sw_version = AXOL_HUB_FW_VERSION,
                        .hw_version = 1,
                    };
                    return tud_control_xfer(rhport, request, (uint8_t *)&cfg, sizeof(cfg));
                }
                return false;

            case GS_USB_BREQ_BT_CONST:
                if (request->bmRequestType_bit.direction == TUSB_DIR_IN) {
                    if (channel >= NUM_CHANNELS)
                        return false;
                    uint32_t fclk = clock_get_hz(clk_sys);
                    struct gs_device_bt_const bt_const = {
                        .feature = 0,
                        .fclk_can = fclk,
                        .tseg1_min = 1, .tseg1_max = 16,
                        .tseg2_min = 1, .tseg2_max = 8,
                        .sjw_max = 4,
                        .brp_min = 1, .brp_max = 256, .brp_inc = 1,
                    };
                    return tud_control_xfer(rhport, request, (uint8_t *)&bt_const, sizeof(bt_const));
                }
                return false;

            case GS_USB_BREQ_BITTIMING:
                if (request->bmRequestType_bit.direction == TUSB_DIR_OUT && wLength <= sizeof(g_control_buffer)) {
                    if (channel >= NUM_CHANNELS)
                        return false;
                    return tud_control_xfer(rhport, request, g_control_buffer, wLength);
                }
                return false;

            case GS_USB_BREQ_MODE:
                if (request->bmRequestType_bit.direction == TUSB_DIR_OUT && wLength <= sizeof(g_control_buffer)) {
                    if (channel >= NUM_CHANNELS)
                        return false;
                    return tud_control_xfer(rhport, request, g_control_buffer, wLength);
                }
                return false;

            case AXOL_BREQ_BOOTLOADER:
                if (request->bmRequestType_bit.direction == TUSB_DIR_OUT && wLength == 0)
                    return tud_control_status(rhport, request);
                return false;

            case AXOL_BREQ_RESET_INFO:
                if (request->bmRequestType_bit.direction == TUSB_DIR_IN) {
                    g_reset_info = (struct axol_reset_info){
                        .version = AXOL_RESET_INFO_VERSION,
                        .chip_reset = g_boot_chip_reset,
                        .watchdog_reason = g_boot_watchdog_reason,
                        .bod = powman_hw->bod,
                        .uptime_us = time_us_64(),
                    };
                    uint16_t len = wLength < sizeof(g_reset_info) ? wLength : sizeof(g_reset_info);
                    return tud_control_xfer(rhport, request, &g_reset_info, len);
                }
                return false;

            default:
                return false;
        }
    }
    else if (stage == CONTROL_STAGE_DATA) {
        if (request->bmRequestType_bit.direction == TUSB_DIR_IN)
            return true;

        if (request->bmRequestType_bit.direction == TUSB_DIR_OUT) {
            switch (breq) {
                case GS_USB_BREQ_HOST_FORMAT:
                    if (wLength == sizeof(struct gs_host_config)) {
                        struct gs_host_config *cfg = (struct gs_host_config *)g_control_buffer;
                        g_host_byte_order = cfg->byte_order;
                        return true;
                    }
                    return false;

                case GS_USB_BREQ_BITTIMING:
                    if (wLength == sizeof(struct gs_device_bittiming)) {
                        struct gs_device_bittiming *bt = (struct gs_device_bittiming *)g_control_buffer;
                        g_channels[channel].bitrate = gs_compute_bitrate(bt, clock_get_hz(clk_sys));
                        return true;
                    }
                    return false;

                case GS_USB_BREQ_MODE:
                    if (wLength == sizeof(struct gs_device_mode)) {
                        struct gs_device_mode *mode = (struct gs_device_mode *)g_control_buffer;
                        struct can_channel_state *ch = &g_channels[channel];
                        // Signal core1 to start or stop — core1 owns all can2040 calls.
                        // Don't gate on ch->started here: core1 may not have processed
                        // an earlier request yet, so started can be stale.  Core1 checks
                        // the real state when it handles the flag.
                        if (mode->mode == GS_CAN_MODE_START) {
                            __dmb();  // ensure bitrate write is visible to core1 before pending_start
                            ch->pending_start = true;
                            __sev();  // wake core1 — it may be idle in __wfe() with no CAN IRQs
                        } else if (mode->mode == GS_CAN_MODE_RESET) {
                            // Invalidate all frames/echoes queued under the old
                            // session before signalling the stop, so nothing
                            // stale can replay after the next START.
                            ch->epoch++;
                            // Cancel any unprocessed START: core1 handles stop
                            // before start, so leaving both pending would end
                            // with the channel running and the RESET lost.
                            ch->pending_start = false;
                            ch->pending_stop = true;
                            __sev();  // wake core1 — a quiet bus generates no IRQs to wake it
                        }
                        return true;
                    }
                    return false;

                default:
                    return false;
            }
        }
    }
    else if (stage == CONTROL_STAGE_ACK) {
        if (breq == AXOL_BREQ_BOOTLOADER)
            reset_usb_boot(0, 0);  // does not return
        return true;
    }

    return false;
}

//--------------------------------------------------------------------
// USB Bulk transfer handlers (run on core0)
//--------------------------------------------------------------------

// Read one frame from host, push to TX queue for CAN transmission,
// and immediately echo back to Linux so tx_context slots are freed
// without waiting for NOTIFY_TX.  This matches CANable's behaviour and
// prevents the deferred-echo latency from exhausting GS_MAX_TX_URBS=10
// slots at high frame rates, which was blocking Python's asyncio loop.
static bool process_one_bulk_out(void) {
    if (tud_vendor_available() < sizeof(struct gs_host_frame))
        return false;

    // Backpressure: don't read from USB unless BOTH the TX queue and the
    // echo queue have space.  An echo must never be dropped: the Linux
    // gs_usb driver only frees a tx_context slot (it has just 10) when the
    // echo returns, so a lost echo leaks the slot permanently and after 10
    // losses the interface's TX path is dead until replug.
    uint32_t push = g_tx_queue.push_pos;
    if ((push - g_tx_queue.pull_pos) >= TX_QUEUE_SIZE)
        return false;
    uint32_t ep = g_echo_queue.push_pos;
    if ((ep - g_echo_queue.pull_pos) >= TX_QUEUE_SIZE)
        return false;

    struct gs_host_frame frame;
    if (tud_vendor_read((uint8_t *)&frame, sizeof(frame)) != sizeof(frame))
        return false;

    uint32_t epoch = (frame.channel < NUM_CHANNELS) ? g_channels[frame.channel].epoch : 0;

    // Push to TX queue for core1 to transmit on CAN bus
    struct tx_queue_entry *entry = &g_tx_queue.entries[push % TX_QUEUE_SIZE];
    entry->host_frame = frame;
    entry->epoch = epoch;
    entry->enqueued_us = time_us_32();
    __dmb();
    g_tx_queue.push_pos = push + 1;
    __sev();  // wake core1 if it called __wfe() with an empty queue

    // Immediately echo back to Linux — don't wait for NOTIFY_TX.
    // Space was verified above, so this never drops.
    g_echo_queue.entries[ep % TX_QUEUE_SIZE].frame = frame;
    g_echo_queue.entries[ep % TX_QUEUE_SIZE].epoch = epoch;
    __dmb();
    g_echo_queue.push_pos = ep + 1;

    return true;
}

// Drain TX echoes from echo queue to USB host.
//
// We send ONE frame per USB IN transfer (at most 20 bytes = one gs_host_frame).
// The Linux gs_usb driver allocates 20-byte URBs.  If the device sends a full
// 64-byte USB packet, the host controller delivers only the first 20 bytes to
// the URB and discards the remaining 44 — exactly 2 out of every 3.2 frames
// are silently lost.  Keeping the ring-FIFO at most 20 bytes before flush()
// produces a "short packet" (20 < 64 = max packet size) that Linux handles
// correctly, one URB per frame.
static void process_echo_queue(void) {
    while (g_echo_queue.push_pos != g_echo_queue.pull_pos) {
        // Only write when the ring FIFO is completely empty.  If there are
        // already bytes queued, the vendord_xfer_cb completion path will
        // drain them and restart the pipeline via the next tud_task() call.
        if (tud_vendor_write_available() < CFG_TUD_VENDOR_TX_BUFSIZE)
            break;

        uint32_t pull = g_echo_queue.pull_pos;
        struct echo_queue_entry *entry = &g_echo_queue.entries[pull % TX_QUEUE_SIZE];
        struct gs_host_frame *frame = &entry->frame;

        // Drop echoes from a closed session: the host freed its tx_context
        // slots at channel RESET, so delivering these would only produce
        // "unexpected echo_id" noise (or misattribute a stale slot).
        if (frame->channel < NUM_CHANNELS &&
            entry->epoch != g_channels[frame->channel].epoch) {
            g_echo_queue.pull_pos = pull + 1;
            continue;
        }

        if (tud_vendor_write((uint8_t *)frame, sizeof(*frame)) == 0)
            break;

        g_echo_queue.pull_pos = pull + 1;
        // Flush immediately: if the endpoint is free this drains the 20 bytes
        // into ep_buf and starts a short-packet USB transfer right away.
        // If the endpoint is busy, the 20 bytes stay in the ring FIFO and the
        // check above will break on the next loop iteration.
        tud_vendor_write_flush();
    }
}

// Forward all queued CAN RX frames to host.
// Same per-frame short-packet strategy as process_echo_queue().
static void process_bulk_in(void) {
    while (g_rx_queue.push_pos != g_rx_queue.pull_pos) {
        __dmb();  // acquire: read entry data only after observing core1's push_pos
        if (tud_vendor_write_available() < CFG_TUD_VENDOR_TX_BUFSIZE)
            break;

        uint32_t pull = g_rx_queue.pull_pos;
        struct rx_queue_entry *entry = &g_rx_queue.entries[pull % RX_QUEUE_SIZE];

        uint8_t dlc = entry->msg.dlc > 8 ? 8 : entry->msg.dlc;
        struct gs_host_frame frame = {
            .echo_id  = GS_HOST_FRAME_ECHO_ID_RX,
            .can_id   = entry->msg.id,
            .can_dlc  = dlc,
            .channel  = entry->channel,
            .flags    = 0,
            .reserved = 0,
        };
        memcpy(frame.data, entry->msg.data, dlc);

        if (tud_vendor_write((uint8_t *)&frame, sizeof(frame)) == 0)
            break;

        g_rx_queue.pull_pos = pull + 1;
        tud_vendor_write_flush();
    }
}

//--------------------------------------------------------------------
// Core0: USB owner
//--------------------------------------------------------------------

// End the host session: everything queued under it — frames not yet on the
// wire, echoes and RX frames not yet delivered — is discarded, and both
// channels are stopped and wiped until the next host opens them.  Core0 only
// (tud_task context or the main loop), never from an ISR.
//
// A USB bus reset must end the session too, not just an unplug: the robot
// host's purge after a dead bus (e-stop) is a USB reset of this device, and
// TinyUSB's BUS_RESET handling never calls tud_umount_cb.  Without this the
// old session's echoes reached the re-enumerated driver ("Unexpected unused
// echo id") and its frames stayed queued, still valid, behind the dead bus.
static void end_usb_session(void) {
    // Echo queue is entirely core0-owned — safe to reset directly.
    g_echo_queue.pull_pos = g_echo_queue.push_pos;

    // Do NOT touch g_tx_queue positions here: core1 may be mid-drain with a
    // captured push watermark, and rewinding push_pos could leave pull_pos
    // ahead of push_pos — making the queue appear permanently full.  The
    // epoch bump below makes core1 discard every queued frame instead.

    // Discard buffered RX frames so they are not sent to the new session.
    __dmb();  // ensure we read the latest push_pos written by core1 IRQ
    g_rx_queue.pull_pos = g_rx_queue.push_pos;

    // Ask core1 to stop both CAN channels.  Set pending_stop unconditionally:
    // started is core1-written and may be stale here (e.g. a START still
    // pending), and core1 no-ops the stop if the channel isn't running.
    // Also cancel any unprocessed START so a channel can't come up after
    // the host has disconnected.
    for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
        g_channels[i].epoch++;  // invalidate queued frames from the dead session
        g_channels[i].pending_start = false;
        g_channels[i].pending_stop = true;
    }
    __sev();  // wake core1 immediately to process pending_stop
}

// Set from the USB ISR on a bus reset, consumed by the core0 main loop.
static volatile bool g_usb_bus_reset;

// Invoked from the USB ISR for every device event.  Only flags the reset:
// end_usb_session() touches core0-owned queue state and must not run here.
void tud_event_hook_cb(uint8_t rhport, uint32_t eventid, bool in_isr) {
    (void)rhport;
    (void)in_isr;
    if (eventid == DCD_EVENT_BUS_RESET)
        g_usb_bus_reset = true;
}

// Invoked when USB host disconnects (or deconfigures the device).
void tud_umount_cb(void) {
    end_usb_session();
}

// Invoked when the host configures the device.  A new configuration is a new
// host session; nothing from before it may survive (covers re-enumerations
// that did not go through a bus reset this firmware saw).
void tud_mount_cb(void) {
    end_usb_session();
}

int main(void) {
    // Capture reset provenance before anything else runs.
    g_boot_chip_reset = powman_hw->chip_reset;
    g_boot_watchdog_reason = watchdog_hw->reason;

    gpio_init(STATUS_LED_GPIO);
    gpio_set_dir(STATUS_LED_GPIO, GPIO_OUT);
    gpio_put(STATUS_LED_GPIO, 1);

    usb_descriptors_init();
    tusb_init();

    // Launch core1 and wait for it to finish CAN setup
    multicore_launch_core1(core1_main);
    multicore_fifo_pop_blocking();  // core1 signals ready

    uint32_t last_wake_us = time_us_32();
    while (true) {
        if (g_usb_bus_reset) {
            g_usb_bus_reset = false;
            end_usb_session();
        }

        tud_task();

        // Core1 sleeps in __wfe() between events.  The stall recovery and the
        // frame age limit must still run on a dead bus even if the engine
        // stops raising irqs, so never let it sleep longer than a few ms.
        uint32_t now = time_us_32();
        if (now - last_wake_us >= 2000) {
            last_wake_us = now;
            __sev();
        }

        if (tud_vendor_mounted()) {
            while (process_one_bulk_out())
                ;
            process_echo_queue();  // immediate TX echoes → host
            process_bulk_in();     // CAN RX frames → host
        }
    }

    return 0;
}
