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

    unsigned int getVaoForPointsAndLines() const { return _vaoForPointsAndLines; }
    unsigned int getVaoForTriangles() const { return _vaoForTriangles; }
    unsigned int getVbo() const { return _vbo; }
    unsigned int getEboForLines() const { return _eboForLines; }
    unsigned int getEboForTriangles() const { return _eboForTriangles; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vbo = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    NumRenderObjects _numObjects;

    _GeometryBuffers() = default;
};
