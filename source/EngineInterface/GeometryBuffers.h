#pragma once

#include <cstdint>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t vertices;
    uint64_t lineIndices;
    uint64_t triangleIndices;
    uint64_t energyParticles;
};

struct CellVertexData
{
    float pos[3];    // x, y, z position (z used for lighting)
    float color[3];  // r, g, b color
};

struct EnergyParticleVertexData
{
    float pos[3];    // x, y, z position
    float color[3];  // r, g, b color
};

class _GeometryBuffers
{
public:
    static GeometryBuffers create();

    unsigned int getVaoForPointsAndLines() const { return _vaoForPointsAndLines; }
    unsigned int getVaoForTriangles() const { return _vaoForTriangles; }
    unsigned int getVaoForEnergyParticles() const { return _vaoForEnergyParticles; }
    unsigned int getVbo() const { return _vbo; }
    unsigned int getVboForEnergyParticles() const { return _vboForEnergyParticles; }
    unsigned int getEboForLines() const { return _eboForLines; }
    unsigned int getEboForTriangles() const { return _eboForTriangles; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vaoForEnergyParticles = 0;
    unsigned int _vbo = 0;
    unsigned int _vboForEnergyParticles = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _energyParticleBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    NumRenderObjects _numObjects;

    _GeometryBuffers() = default;
};
