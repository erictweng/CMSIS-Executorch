// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "fs_api.h"
#include "fs_detect.h"  // fs_box_t

#if defined(FINGERSPELL_LCD)
// rgb is the same 128x128 RGB888 crop passed to inference.
bool fs_display_result(const uint8_t* rgb, char predicted, float confidence,
                       char expected, bool pass);
// Borrow a persistent RGB888 crop for the next fs_ui_render(); no image copy.
void fs_display_set_frame(const uint8_t* rgb);
// The model's top-3 letters and probabilities for the next fs_ui_render() (fs_decoder_t only
// carries the top letter, and only above the commit threshold).
void fs_display_set_top3(const char letters[3], const float probs[3]);
// The full 320x320 RGB565 camera frame (borrowed until the next fs_ui_render()) for the
// small inset, and the hand box on it in frame pixels: hand false draws the inset with a
// red frame and no box, fresh false (a held box) draws it muted instead of green.
void fs_display_set_detection(const uint16_t* frame565, bool hand, bool fresh, float x0, float y0, float x1,
                              float y1);
// The 21 hand landmarks for the next fs_ui_render(), in frame pixels, with the hand box the
// crop view shows (the skeleton is drawn on the crop view through that box and as dots on
// the inset); NULL = none.
void fs_display_set_landmarks(const fs_box_t* box, const float xyz[21][3]);
// The hearing person's last spoken reply (keyword spotting), shown as "> WORD" on the last row
// of the words card from the next fs_ui_render() on; NULL or "" removes the row.
void fs_display_set_reply(const char* word);
#endif
