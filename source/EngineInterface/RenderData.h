#pragma once

#include <cstdint>

struct RenderBuffers
{
    unsigned int vboForPoints = 0;
    unsigned int eboForLines = 0;
};

struct NumRenderObjects
{
    uint64_t vertices;
    uint64_t lineIndices;
};

struct VertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    float padding;
};
