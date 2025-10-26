#pragma once

#include <cstdint>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t vertices;
    uint64_t lineIndices;
    uint64_t triangleIndices;
    uint64_t energyParticles;
    uint64_t locations;
    uint64_t selectedObjects;
    uint64_t connectionArrowVertices;
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

struct LocationVertexData
{
    float pos[2];       // x, y position
    float color[3];     // r, g, b color
    int shapeType;      // 0 = circular, 1 = rectangular
    float dimension1;   // radius for circular, width for rectangular
    float dimension2;   // unused for circular, height for rectangular
    float fadeoutRadius; // fadeout radius for the location
    float opacity;      // opacity/transparency of the location
};

struct SelectedObjectVertexData
{
    float pos[2];       // x, y position
};

struct ConnectionArrowVertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    int arrowFlags;  // bit 0: arrow to first vertex, bit 1: arrow to second vertex
};

class _GeometryBuffers
{
public:
    static GeometryBuffers create();

    unsigned int getVaoForPointsAndLines() const { return _vaoForPointsAndLines; }
    unsigned int getVaoForTriangles() const { return _vaoForTriangles; }
    unsigned int getVaoForEnergyParticles() const { return _vaoForEnergyParticles; }
    unsigned int getVaoForLocations() const { return _vaoForLocations; }
    unsigned int getVaoForSelectedObjects() const { return _vaoForSelectedObjects; }
    unsigned int getVaoForSelectedConnections() const { return _vaoForSelectedConnections; }
    unsigned int getVboForCells() const { return _vboForCells; }
    unsigned int getVboForEnergyParticles() const { return _vboForEnergyParticles; }
    unsigned int getVboForLocations() const { return _vboForLocations; }
    unsigned int getVboForSelectedObjects() const { return _vboForSelectedObjects; }
    unsigned int getVboForSelectedConnections() const { return _vboForSelectedConnections; }
    unsigned int getEboForLines() const { return _eboForLines; }
    unsigned int getEboForTriangles() const { return _eboForTriangles; }

    void resizeIfNecessary(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vaoForEnergyParticles = 0;
    unsigned int _vaoForLocations = 0;
    unsigned int _vaoForSelectedObjects = 0;
    unsigned int _vaoForSelectedConnections = 0;
    unsigned int _vboForCells = 0;
    unsigned int _vboForEnergyParticles = 0;
    unsigned int _vboForLocations = 0;
    unsigned int _vboForSelectedObjects = 0;
    unsigned int _vboForSelectedConnections = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _energyParticleBufferCapacity = 0;
    uint64_t _locationBufferCapacity = 0;
    uint64_t _selectedObjectBufferCapacity = 0;
    uint64_t _connectionArrowVertexBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    NumRenderObjects _numObjects;

    _GeometryBuffers() = default;
};
