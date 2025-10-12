#pragma once

#include <cstdint>

struct RenderBuffers
{
    uint64_t vertexBufferCapacity = 0;
    uint64_t lineIndexBufferCapacity = 0;

    unsigned int vboForPoints = 0;
    unsigned int vaoForLines = 0;
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
