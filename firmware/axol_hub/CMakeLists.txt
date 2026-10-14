# Generated Cmake Pico project file

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 13_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================
set(PICO_BOARD pico2 CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)

project(axol_hub C CXX ASM)

# Firmware version, BCD-coded (0x0200 = 2.00).  Reported as the USB bcdDevice
# and gs_usb sw_version; almond_axol.hub flashes any hub reporting an older
# one.  Bump it with every firmware change, rebuild, and copy build/axol_hub.uf2
# to almond_axol/hub/axol_hub.uf2 (see README.md).
set(AXOL_HUB_FW_VERSION 0x0200)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()

# External libraries
include(FetchContent)

# can2040
FetchContent_Declare(
    can2040_repo
    GIT_REPOSITORY https://github.com/KevinOConnor/can2040.git
    GIT_TAG        2988d4f11d8bff93f5a3d317fcd5e384a6aa3481
)

FetchContent_MakeAvailable(can2040_repo)

# Add executable. Default name is the project name, version 0.1
add_executable(axol_hub
    firmware.c
    usb_descriptors.c
    ${can2040_repo_SOURCE_DIR}/src/can2040.c
)

pico_set_program_name(axol_hub "axol_hub")
pico_set_program_version(axol_hub "${AXOL_HUB_FW_VERSION}")

# Both stdio backends intentionally disabled — no UART or USB-CDC logging
pico_enable_stdio_uart(axol_hub 0)
pico_enable_stdio_usb(axol_hub 0)

# Add the standard library to the build
target_link_libraries(axol_hub
        pico_stdlib
        pico_multicore
        hardware_clocks
        pico_unique_id
        tinyusb_device
)

# Add the standard include files to the build
target_include_directories(axol_hub PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${can2040_repo_SOURCE_DIR}/src
)

# TinyUSB headers (tusb.h); required when using custom tusb_config.h
if(PICO_SDK_PATH)
    target_include_directories(axol_hub PRIVATE ${PICO_SDK_PATH}/lib/tinyusb/src)
endif()

# Define TinyUSB config header location
target_compile_definitions(axol_hub PRIVATE
        CFG_TUSB_CONFIG_FILE=\"tusb_config.h\"
        AXOL_HUB_FW_VERSION=${AXOL_HUB_FW_VERSION}
)

pico_add_extra_outputs(axol_hub)

