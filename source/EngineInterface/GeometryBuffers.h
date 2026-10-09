#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "Definitions.h"

struct NumRenderObjects
{
    uint64_t objects;
    uint64_t fluidParticles;
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
    float pos[3];              // x, y, z position (z used for lighting)
    float color[3];            // r, g, b color
    int state;                 // Bit 0..7 = cell type
                               // Bit 8..15 = object type (ObjectType_Solid, ObjectType_FreeCell, ObjectType_Cell)
                               // Bit 16 = is isolated (zero connections)
    float highlightIntensity;  // Highlight intensity in [0, 1]
};

struct FluidParticleVertexData
{
    float pos[3];    // x, y, z position
    float color[3];  // r, g, b color
    float glow;      // Glow intensity (0.0 = no glow, 1.0 = full glow)
};

struct LocationVertexData
{
    float pos[2];         // x, y position
    float color[3];       // r, g, b color
    int shapeType;        // 0 = circular, 1 = rectangular
    float dimension1;     // Radius for circular, width for rectangular
    float dimension2;     // Unused for circular, height for rectangular
    float fadeoutRadius;  // Fadeout radius for the location
    float opacity;        // Opacity/transparency of the location
    int fieldType;        // Force field whose height map shades the background (ForceField_None = no shading)
    float fieldParam1;    // Orientation sign (radial), angle (linear), spatial size (Perlin noise)
    float fieldParam2;    // Time coordinate (Perlin noise)
    int colored;          // 0 = the background keeps its color and is only shaded by the force field
};

struct SelectedObjectVertexData
{
    float pos[2];  // x, y position
};

struct ConnectionArrowVertexData
{
    float pos[2];                     // x, y position
    float color[3];                   // r, g, b color
    float connectionWeightToObject1;  // Connection weight for arrow toward first vertex
    float connectionWeightToObject2;  // Connection weight for arrow toward second vertex
};

struct AttackEventVertexData
{
    float pos[2];    // x, y position
    float color[3];  // r, g, b color (red for attacked)
};

struct DetonationEventVertexData
{
    uint64_t objectId;
    float pos[2];  // x, y position
    float radius;  // Detonator radius
};

using GeometryBufferType = int;
enum GeometryBufferType_
{
    GeometryBufferType_Objects,
    GeometryBufferType_FluidParticles,
    GeometryBufferType_Locations,
    GeometryBufferType_SelectedObjects,
    GeometryBufferType_LineIndices,
    GeometryBufferType_TriangleIndices,
    GeometryBufferType_SelectedConnections,
    GeometryBufferType_AttackEvents,
    GeometryBufferType_DetonationEvents,
    GeometryBufferType_Count
};

namespace GeometryBufferLayout
{
    inline constexpr std::array<uint64_t, GeometryBufferType_Count> ElementSizes = {
        sizeof(ObjectVertexData),
        sizeof(FluidParticleVertexData),
        sizeof(LocationVertexData),
        sizeof(SelectedObjectVertexData),
        sizeof(unsigned int),
        sizeof(unsigned int),
        sizeof(ConnectionArrowVertexData),
        sizeof(AttackEventVertexData),
        sizeof(DetonationEventVertexData),
    };

    inline constexpr std::array<uint64_t, GeometryBufferType_Count> MinCapacities = {100000, 100000, 1000, 10000, 100000, 100000, 100000, 10000, 10000};

    uint64_t getNumElements(NumRenderObjects const& numObjects, GeometryBufferType type);
}

// Memory of a geometry buffer that the GPU engine can import and write into directly
struct SharedGeometryMemory
{
    void* win32Handle = nullptr;  // NT handle on Windows, the importer closes it
    int fd = -1;                  // File descriptor on Linux, owned by the importer after a successful import
    uint64_t allocationSize = 0;
    bool dedicatedAllocation = false;
};

// Vertex and index buffers holding the visible part of the simulation for rendering
class _GeometryBuffers
{
public:
    virtual ~_GeometryBuffers() = default;

    // Enlarges the buffers if necessary
    void updateNumObjects(NumRenderObjects const& numRenderObjects);

    NumRenderObjects getNumObjects() const;

    bool hasReallocatedBuffers() const;

    uint64_t getCapacity(GeometryBufferType type) const;

    virtual bool isMemoryShareable() const = 0;
    virtual SharedGeometryMemory shareMemory(GeometryBufferType type) = 0;

    virtual void upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes) = 0;
    virtual void download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const = 0;

    std::vector<ObjectVertexData> getCellData() const;
    std::vector<FluidParticleVertexData> getFluidParticleData() const;
    std::vector<LocationVertexData> getLocationData() const;
    std::vector<SelectedObjectVertexData> getSelectedObjectData() const;
    std::vector<unsigned int> getLineIndices() const;
    std::vector<unsigned int> getTriangleIndices() const;
    std::vector<ConnectionArrowVertexData> getSelectedConnectionData() const;
    std::vector<AttackEventVertexData> getAttackEventData() const;
    std::vector<DetonationEventVertexData> getDetonationEventData() const;

protected:
    virtual void reallocate(GeometryBufferType type, uint64_t sizeInBytes) = 0;

private:
    template <typename T>
    std::vector<T> downloadElements(GeometryBufferType type) const;

    NumRenderObjects _numObjects = {};
    bool _reallocatedBuffers = false;
    std::array<uint64_t, GeometryBufferType_Count> _capacities = {};
};

// Geometry buffers in host memory, e.g. for tests without a graphics device
class _HostGeometryBuffers : public _GeometryBuffers
{
public:
    static GeometryBuffers create();

    bool isMemoryShareable() const override;
    SharedGeometryMemory shareMemory(GeometryBufferType type) override;

    void upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes) override;
    void download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const override;

protected:
    void reallocate(GeometryBufferType type, uint64_t sizeInBytes) override;

private:
    _HostGeometryBuffers() = default;

    std::array<std::vector<uint8_t>, GeometryBufferType_Count> _buffers;
};
