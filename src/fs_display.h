// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "fs_api.h"

#if defined(FINGERSPELL_LCD)
// rgb is the same 128x128 RGB888 crop passed to inference.
bool fs_display_result(const uint8_t* rgb, char predicted, float confidence,
                       char expected, bool pass);
// Borrow a persistent RGB888 crop for the next fs_ui_render(); no image copy.
void fs_display_set_frame(const uint8_t* rgb);
// The model's top-3 letters and probabilities for the next fs_ui_render() (fs_decoder_t only
// carries the top letter, and only above the commit threshold).
void fs_display_set_top3(const char letters[3], const float probs[3]);
#endif
