#pragma once

#include <cmath>
#include <cstdint>

#include "Definitions.h"

// Color derivations shared by the renderer (see getObjectColor in GeometryKernels.cu) and the GUI
class ObjectColoring
{
public:
    HOST_DEVICE static uint32_t hsvToRgb(float h, float s, float v)
    {
        float c = v * s;
        float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
        float m = v - c;
        float r, g, b;
        int hi = static_cast<int>(h * 6.0f) % 6;
        switch (hi) {
        case 0:
            r = c;
            g = x;
            b = 0;
            break;
        case 1:
            r = x;
            g = c;
            b = 0;
            break;
        case 2:
            r = 0;
            g = c;
            b = x;
            break;
        case 3:
            r = 0;
            g = x;
            b = c;
            break;
        case 4:
            r = x;
            g = 0;
            b = c;
            break;
        default:
            r = c;
            g = 0;
            b = x;
            break;
        }
        return (toRgbByte(r + m) << 16) | (toRgbByte(g + m) << 8) | toRgbByte(b + m);
    }

    HOST_DEVICE static void rgbToHsv(float r, float g, float b, float& h, float& s, float& v)
    {
        float maxC = fmaxf(r, fmaxf(g, b));
        float minC = fminf(r, fminf(g, b));
        float delta = maxC - minC;
        h = 0.0f;
        if (delta > 0.0f) {
            if (maxC == r) {
                h = fmodf((g - b) / delta, 6.0f);
            } else if (maxC == g) {
                h = (b - r) / delta + 2.0f;
            } else {
                h = (r - g) / delta + 4.0f;
            }
            h /= 6.0f;
            if (h < 0.0f) {
                h += 1.0f;
            }
        }
        s = maxC <= 0.0f ? 0.0f : delta / maxC;
        v = maxC;
    }

    // Identifying color of a lineage or creature, derived from its id
    HOST_DEVICE static uint32_t getColorFromId(uint32_t id)
    {
        uint32_t hash1 = id * 2654435761u;
        uint32_t hash2 = id * 2246822519u;
        uint32_t hash3 = id * 3266489917u;
        float h = static_cast<float>(hash1 & 0xFFFFu) / 65535.0f;
        float s = 0.7f + 0.3f * (static_cast<float>(hash2 & 0xFFFFu) / 65535.0f);
        float v = 0.6f + 0.1f * (static_cast<float>(hash3 & 0xFFFFu) / 65535.0f);
        return hsvToRgb(h, s, v);
    }

private:
    HOST_DEVICE static uint32_t toRgbByte(float value)
    {
        auto result = static_cast<int>(value * 255.0f + 0.5f);
        if (result < 0) {
            result = 0;
        } else if (result > 255) {
            result = 255;
        }
        return static_cast<uint32_t>(result);
    }
};
