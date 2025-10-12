#pragma once

#include <cstdint>

#include "Definitions.h"

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

class _GeometryBuffers
{
public:
    static GeometryBuffers create();

    unsigned int getVao() const { return vao; }
    unsigned int getVbo() const { return vbo; }
    unsigned int getEbo() const { return ebo; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

private:
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;

    uint64_t vertexBufferCapacity = 0;
    uint64_t lineIndexBufferCapacity = 0;

    _GeometryBuffers() = default;
};
