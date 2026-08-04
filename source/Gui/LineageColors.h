#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <imgui.h>

#include <Base/Definitions.h>

#include <EngineInterface/ObjectColoring.h>

namespace Const
{
    auto constexpr LineageHueShift = 0.03f;    // Hue offset between consecutive plot rows of a lineage
    auto constexpr LineageBrightening = 1.5f;  // The identifying colors of the renderer are too dark on the dark GUI background
    auto constexpr LineageSaturationFactor = 0.85f;
}

// Identifying color of a lineage as used by the renderer, brightened for the dark GUI background;
// hueShiftSteps shifts the hue slightly so that consecutive plot rows of a lineage remain distinguishable
inline ImColor getLineageColor(int64_t lineageId, int hueShiftSteps = 0)
{
    auto rgb = ObjectColoring::getColorFromId(toUInt32(lineageId));
    float h, s, v;
    ObjectColoring::rgbToHsv(toFloat((rgb >> 16) & 0xff) / 255.0f, toFloat((rgb >> 8) & 0xff) / 255.0f, toFloat(rgb & 0xff) / 255.0f, h, s, v);
    h = std::fmod(h + toFloat(hueShiftSteps) * Const::LineageHueShift, 1.0f);
    return ImColor::HSV(h, s * Const::LineageSaturationFactor, std::min(1.0f, v * Const::LineageBrightening));
}
