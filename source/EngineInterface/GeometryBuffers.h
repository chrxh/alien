#pragma once

#include <cstdint>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t vertices;
    uint64_t lineIndices;
    uint64_t triangleIndices;
};

struct VertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    float zPos;      // z position for lighting (was padding)
};

class _GeometryBuffers
{
public:
    static GeometryBuffers create();

    unsigned int getVao() const { return _vao; }
    unsigned int getVbo() const { return _vbo; }
    unsigned int getEbo() const { return _ebo; }
    unsigned int getTbo() const { return _tbo; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

private:
    unsigned int _vao = 0;
    unsigned int _vbo = 0;
    unsigned int _ebo = 0;
    unsigned int _tbo = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    _GeometryBuffers() = default;
};
