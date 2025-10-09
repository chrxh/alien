#pragma once

#include <cstdint>

struct RenderBuffers
{
    unsigned int vboForPoints = 0;
};

struct NumRenderObjects
{
    uint64_t vertices;
};

struct VertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    float padding;
};
