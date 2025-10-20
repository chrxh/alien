#pragma once

#include <cstdint>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t vertices;
    uint64_t lineIndices;
    uint64_t triangleIndices;
    uint64_t energyParticles;
    uint64_t zones;
};

struct CellVertexData
{
    float pos[3];    // x, y, z position (z used for lighting)
    float color[3];  // r, g, b color
    int state;       // = 1 (signal active)
};

struct EnergyParticleVertexData
{
    float pos[3];    // x, y, z position
    float color[3];  // r, g, b color
};

struct ZoneVertexData
{
    float pos[2];       // x, y position
    float color[3];     // r, g, b color
    int shapeType;      // 0 = circular, 1 = rectangular
    float dimension1;   // radius for circular, width for rectangular
    float dimension2;   // unused for circular, height for rectangular
    float fadeoutRadius; // fadeout radius for the zone
    float opacity;      // opacity/transparency of the zone
};

class _GeometryBuffers
{
public:
    static GeometryBuffers create();

    unsigned int getVaoForPointsAndLines() const { return _vaoForPointsAndLines; }
    unsigned int getVaoForTriangles() const { return _vaoForTriangles; }
    unsigned int getVaoForEnergyParticles() const { return _vaoForEnergyParticles; }
    unsigned int getVaoForZones() const { return _vaoForZones; }
    unsigned int getVboForCells() const { return _vboForCells; }
    unsigned int getVboForEnergyParticles() const { return _vboForEnergyParticles; }
    unsigned int getVboForZones() const { return _vboForZones; }
    unsigned int getEboForLines() const { return _eboForLines; }
    unsigned int getEboForTriangles() const { return _eboForTriangles; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vaoForEnergyParticles = 0;
    unsigned int _vaoForZones = 0;
    unsigned int _vboForCells = 0;
    unsigned int _vboForEnergyParticles = 0;
    unsigned int _vboForZones = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _energyParticleBufferCapacity = 0;
    uint64_t _zoneBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    NumRenderObjects _numObjects;

    _GeometryBuffers() = default;
};
