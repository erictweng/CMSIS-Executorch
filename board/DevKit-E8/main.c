/*---------------------------------------------------------------------------
 * Copyright (c) 2025-2026 Arm Limited (or its affiliates). All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * DevKit-E8 (M55_HP) board bring-up for a headless Ethos-U85 runner. Derived
 * from the pack's Boards/DevKit-e8/Layers/M55_HP/main.c without the MIPI,
 * USB, Ethernet and VIO initialisation.
 *---------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include CMSIS_device_header

#include "board_config.h"
#include "main.h"

#include "se_services_port.h"

#if defined(RTE_Drivers_CPI)
#include "cpi.h"
#include "sys_ctrl_cpi.h"

/*
 * A core reset (J-Link, debugger) does not reset the camera controller: a capture
 * left running keeps DMA-ing frames into SRAM, and with that going on the Secure
 * Enclave handshake below hung until a power cycle. Stop it first. Only touch the
 * CPI when its clock is on: accessing it with the clock off locks the bus.
 */
static void stop_leftover_camera_capture(void)
{
    if ((CLKCTL_PER_MST->PERIPH_CLK_ENA & PERIPH_CLK_ENA_CPI_CKEN) == 0U) {
        return; /* not started since power-up */
    }
    CPI->CAM_INTR_ENA = 0U;
    cpi_stop_capture(CPI);
    for (uint32_t i = 0U; i < 10000000U; i++) { /* the frame in flight ends within ~35 ms */
        if (cpi_get_capture_status(CPI) == CPI_VIDEO_CAPTURE_STATUS_NOT_CAPTURING) {
            break;
        }
    }
}
#endif

int main(void)
{
#if defined(RTE_Drivers_CPI)
    stop_leftover_camera_capture();
#endif

    /* Apply the Conductor pin configuration (includes the UART4 console pins) */
    board_pins_config();

    /* Apply the Conductor GPIO configuration */
    board_gpios_config();

    /* Bring up the Secure Enclave services (MHU link to the SE) */
    se_services_port_init();

    /* Request the clocks the SE has to enable for this core */
    board_clocks_config(CLKEN_HFOSC_MASK | CLKEN_CLK_100M_MASK);

    /* Initialize STDIO (UART4 on the PRG USB connector) */
    stdio_init();

    #if defined(ETHOSU_ARCH)
    /* Initialize Ethos NPU */
    ethos_setup();
    #endif

    return app_main();
}
