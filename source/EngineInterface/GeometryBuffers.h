#pragma once

#include <cstdint>
#include <vector>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t objects;
    uint64_t blurryParticles;
    uint64_t locations;
    uint64_t lineIndices;

    uint64_t triangleIndices;
    uint64_t selectedObjects;
    uint64_t connectionArrowVertices;
    uint64_t attackEventVertices;
    uint64_t detonationEventVertices;
};

struct ObjectVertexData
{
    float pos[3];         // x, y, z position (z used for lighting)
    float color[3];       // r, g, b color
    int state;            // Bit 0..7 = cell type
                          // Bit 8..15 = object type (ObjectType_Structure, ObjectType_FreeCell, ObjectType_Cell)
                          // Bit 16 = occurrence in triangle or quad
    float signalChanges;  // signal changes in [0, 1]
};

struct BlurryParticleVertexData
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
    float pos[2];  // x, y position
};

struct ConnectionArrowVertexData
{
    float pos[2];                     // x, y position
    float color[3];                   // r, g, b color
    float connectionWeightToObject1;  // connection weight for arrow toward first vertex
    float connectionWeightToObject2;  // connection weight for arrow toward second vertex
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
    unsigned int getVaoForBlurryParticles() const { return _vaoForBlurryParticles; }
    unsigned int getVaoForLocations() const { return _vaoForLocations; }
    unsigned int getVaoForSelectedObjects() const { return _vaoForSelectedObjects; }
    unsigned int getVaoForSelectedConnections() const { return _vaoForSelectedConnections; }
    unsigned int getVaoForAttackEvents() const { return _vaoForAttackEvents; }
    unsigned int getVaoForDetonationEvents() const { return _vaoForDetonationEvents; }
    unsigned int getVboForObjects() const { return _vboForObjects; }
    unsigned int getVboForBlurryParticles() const { return _vboForBlurryParticles; }
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
    void setCellData(ObjectVertexData const* data, uint64_t count);
    void setBlurryParticleData(BlurryParticleVertexData const* data, uint64_t count);
    void setLocationData(LocationVertexData const* data, uint64_t count);
    void setSelectedObjectData(SelectedObjectVertexData const* data, uint64_t count);
    void setLineIndices(unsigned int const* data, uint64_t count);
    void setTriangleIndices(unsigned int const* data, uint64_t count);
    void setSelectedConnectionData(ConnectionArrowVertexData const* data, uint64_t count);
    void setAttackEventData(AttackEventVertexData const* data, uint64_t count);
    void setDetonationEventData(DetonationEventVertexData const* data, uint64_t count);

    // Methods for downloading data from OpenGL buffers to host memory (for tests)
    std::vector<ObjectVertexData> getCellData() const;
    std::vector<BlurryParticleVertexData> getBlurryParticleData() const;
    std::vector<LocationVertexData> getLocationData() const;
    std::vector<SelectedObjectVertexData> getSelectedObjectData() const;
    std::vector<unsigned int> getLineIndices() const;
    std::vector<unsigned int> getTriangleIndices() const;
    std::vector<ConnectionArrowVertexData> getSelectedConnectionData() const;
    std::vector<AttackEventVertexData> getAttackEventData() const;
    std::vector<DetonationEventVertexData> getDetonationEventData() const;

private:
    unsigned int _vaoForPointsAndLines = 0;
    unsigned int _vaoForTriangles = 0;
    unsigned int _vaoForBlurryParticles = 0;
    unsigned int _vaoForLocations = 0;
    unsigned int _vaoForSelectedObjects = 0;
    unsigned int _vaoForSelectedConnections = 0;
    unsigned int _vaoForAttackEvents = 0;
    unsigned int _vaoForDetonationEvents = 0;
    unsigned int _vboForObjects = 0;
    unsigned int _vboForBlurryParticles = 0;
    unsigned int _vboForLocations = 0;
    unsigned int _vboForSelectedObjects = 0;
    unsigned int _vboForSelectedConnections = 0;
    unsigned int _vboForAttackEvents = 0;
    unsigned int _vboForDetonationEvents = 0;
    unsigned int _eboForLines = 0;
    unsigned int _eboForTriangles = 0;

    uint64_t _vertexBufferCapacity = 0;
    uint64_t _blurryParticleBufferCapacity = 0;
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
