#pragma once

#include <cstdint>
#include <vector>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t cells;
    uint64_t energyParticles;
    uint64_t locations;
    uint64_t lineIndices;

    uint64_t triangleIndices;
    uint64_t selectedObjects;
    uint64_t connectionArrowVertices;
    uint64_t attackEventVertices;
    uint64_t detonationEventVertices;
};

struct CellVertexData
{
    float pos[3];    // x, y, z position (z used for lighting)
    float color[3];  // r, g, b color
    int state;       // Bit 0..7 = signal state (1= highlighted, 2 = strongly highlighted),
                     // Bit 8..15 = cell type
                     // Bit 16 = occurrence in triangle or quad
};

struct EnergyParticleVertexData
{
    float pos[3];    // x, y, z position
    float color[3];  // r, g, b color
};

struct LocationVertexData
{
    float pos[2];         // x, y position
    float color[3];       // r, g, b color
    int shapeType;        // 0 = circular, 1 = rectangular
    float dimension1;     // radius for circular, width for rectangular
    float dimension2;     // unused for circular, height for rectangular
    float fadeoutRadius;  // fadeout radius for the location
    float opacity;        // opacity/transparency of the location
};

struct SelectedObjectVertexData
{
    float pos[2];              // x, y position
    int hasSignalRestriction;  // 1 if signal restriction is active, 0 otherwise
    float startAngle;          // start angle in degrees (0-360)
    float endAngle;            // end angle in degrees (0-360)
};

struct ConnectionArrowVertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color
    int arrowFlags;  // bit 0: arrow to first vertex, bit 1: arrow to second vertex
};

struct AttackEventVertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color (red for attacked)
};

struct DetonationEventVertexData
{
    float pos[2];  // x, y position
    float radius;  // circle radius
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
    unsigned int getVaoForAttackEvents() const { return _vaoForAttackEvents; }
    unsigned int getVaoForDetonationEvents() const { return _vaoForDetonationEvents; }
    unsigned int getVboForCells() const { return _vboForCells; }
    unsigned int getVboForEnergyParticles() const { return _vboForEnergyParticles; }
    unsigned int getVboForLocations() const { return _vboForLocations; }
    unsigned int getVboForSelectedObjects() const { return _vboForSelectedObjects; }
    unsigned int getVboForSelectedConnections() const { return _vboForSelectedConnections; }
    unsigned int getVboForAttackEvents() const { return _vboForAttackEvents; }
    unsigned int getVboForDetonationEvents() const { return _vboForDetonationEvents; }
    unsigned int getEboForLines() const { return _eboForLines; }
    unsigned int getEboForTriangles() const { return _eboForTriangles; }

    void updateNumObjects(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

    // Methods for uploading data from host memory (used in no-interop mode)
    void uploadCellData(CellVertexData const* data, uint64_t count);
    void uploadEnergyParticleData(EnergyParticleVertexData const* data, uint64_t count);
    void uploadLocationData(LocationVertexData const* data, uint64_t count);
    void uploadSelectedObjectData(SelectedObjectVertexData const* data, uint64_t count);
    void uploadLineIndices(unsigned int const* data, uint64_t count);
    void uploadTriangleIndices(unsigned int const* data, uint64_t count);
    void uploadSelectedConnectionData(ConnectionArrowVertexData const* data, uint64_t count);
    void uploadAttackEventData(AttackEventVertexData const* data, uint64_t count);
    void uploadDetonationEventData(DetonationEventVertexData const* data, uint64_t count);

    // Methods for downloading data from OpenGL buffers to host memory (for tests)
    std::vector<CellVertexData> downloadCellData() const;
    std::vector<EnergyParticleVertexData> downloadEnergyParticleData() const;
    std::vector<LocationVertexData> downloadLocationData() const;
    std::vector<SelectedObjectVertexData> downloadSelectedObjectData() const;
    std::vector<unsigned int> downloadLineIndices() const;
    std::vector<unsigned int> downloadTriangleIndices() const;
    std::vector<ConnectionArrowVertexData> downloadSelectedConnectionData() const;
    std::vector<AttackEventVertexData> downloadAttackEventData() const;
    std::vector<DetonationEventVertexData> downloadDetonationEventData() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vaoForEnergyParticles = 0;
    unsigned int _vaoForLocations = 0;
    unsigned int _vaoForSelectedObjects = 0;
    unsigned int _vaoForSelectedConnections = 0;
    unsigned int _vaoForAttackEvents = 0;
    unsigned int _vaoForDetonationEvents = 0;
    unsigned int _vboForCells = 0;
    unsigned int _vboForEnergyParticles = 0;
    unsigned int _vboForLocations = 0;
    unsigned int _vboForSelectedObjects = 0;
    unsigned int _vboForSelectedConnections = 0;
    unsigned int _vboForAttackEvents = 0;
    unsigned int _vboForDetonationEvents = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _energyParticleBufferCapacity = 0;
    uint64_t _locationBufferCapacity = 0;
    uint64_t _selectedObjectBufferCapacity = 0;
    uint64_t _connectionArrowVertexBufferCapacity = 0;
    uint64_t _attackEventVertexBufferCapacity = 0;
    uint64_t _detonationEventVertexBufferCapacity = 0;
    uint64_t _lineIndexBufferCapacity = 0;
    uint64_t _triangleIndexBufferCapacity = 0;

    NumRenderObjects _numObjects;

    _GeometryBuffers() = default;
};
