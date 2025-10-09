#pragma once

struct RenderBuffers
{
    unsigned int vboForPoints = 0;
};

struct NumRenderObjects
{
    uint64_t vertices = 0;
};

struct VertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    float padding;
};
