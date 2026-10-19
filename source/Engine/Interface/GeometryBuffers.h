#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Base/MathTypes.h>

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
    uint64_t shadowGridNodes;
};

struct ObjectVertexData
{
    float pos[3];              // x, y, z position (z used for lighting)
    float color[3];            // r, g, b color
    int state;                 // Bit 0..7 = cell type
                               // Bit 8..15 = object type (ObjectType_Solid, ObjectType_FreeCell, ObjectType_Cell)
                               // Bit 16 = is isolated (zero connections)
    float highlightIntensity;  // Highlight intensity in [0, 1]
    float activity;            // Positive part of signal #0 in [0, 1] or 1 while the cell is attacked, smoothed over time
};

struct FluidParticleVertexData
{
    float pos[3];    // x, y position, z: 0 for energy particles, 1 for fluids
    float color[3];  // r, g, b color
    float glow;      // Glow intensity (0.0 = no glow, 1.0 = full glow)
    float vel[2];    // Velocity in world units per timestep
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

// Visible sizes of the objects close up in the ray traced view. The shadow grid measures the area of the objects outside the extracted region with
// them.
struct RayTracedObjectSizes
{
    static constexpr float CellRadius = 0.34f;
    static constexpr float FreeCellRadius = 0.29f;
    static constexpr float SolidRadius = 0.4f;
    static constexpr float StrandHalfWidth = 0.165f;
};

// Grid that collects the objects outside the extracted region, so that they can shade the light of the radiation sources inside it
struct ShadowGridLayout
{
    RealVector2D origin;
    float nodeDistance = 1.0f;
    IntVector2D size;

    uint64_t getNumNodes() const;
};

// Fixed point sums of the objects around a node of the shadow grid, each object is distributed bilinearly over the four nodes around it. The
// areas include the shares of the objects in their strands and polygons.
struct ShadowGridNodeData
{
    static constexpr float FixedPointScale = 1024.0f;

    uint32_t numFluids;
    uint32_t tissueArea;        // Area of the cells and free cells
    uint32_t tissueHueArea[3];  // Area of the cells and free cells weighted with their r, g, b hue
    uint32_t solidArea;
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
    GeometryBufferType_ShadowGrid,
    GeometryBufferType_Count
};

class GeometryBufferLayout
{
public:
    static constexpr std::array<uint64_t, GeometryBufferType_Count> ElementSizes = {
        sizeof(ObjectVertexData),
        sizeof(FluidParticleVertexData),
        sizeof(LocationVertexData),
        sizeof(SelectedObjectVertexData),
        sizeof(unsigned int),
        sizeof(unsigned int),
        sizeof(ConnectionArrowVertexData),
        sizeof(AttackEventVertexData),
        sizeof(DetonationEventVertexData),
        sizeof(ShadowGridNodeData),
    };

    static constexpr std::array<uint64_t, GeometryBufferType_Count> MinCapacities = {100000, 100000, 1000, 10000, 100000, 100000, 100000, 10000, 10000, 10000};

    static uint64_t getNumElements(NumRenderObjects const& numObjects, GeometryBufferType type);
};

// Memory of a geometry buffer that the GPU engine can import and write into directly
struct SharedGeometryMemory
{
    void* win32Handle = nullptr;  // NT handle on Windows, stays owned by the geometry buffers
    int fd = -1;                  // File descriptor on Linux, owned by the importer after a successful import
    uint64_t allocationSize = 0;
    bool dedicatedAllocation = false;
};

// Vertex and index buffers holding the visible part of the simulation for rendering
class _GeometryBuffers
{
public:
    virtual ~_GeometryBuffers() = default;

    // Enlarges the buffers if necessary. The layout of the shadow grid determines the number of its nodes.
    void updateNumObjects(NumRenderObjects numRenderObjects, ShadowGridLayout const& shadowGridLayout = ShadowGridLayout());

    NumRenderObjects getNumObjects() const;
    ShadowGridLayout const& getShadowGridLayout() const;

    // Changes whenever the buffer gets new memory
    uint64_t getAllocationId(GeometryBufferType type) const;

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
    std::vector<ShadowGridNodeData> getShadowGridData() const;

protected:
    virtual void reallocate(GeometryBufferType type, uint64_t sizeInBytes) = 0;

private:
    template <typename T>
    std::vector<T> downloadElements(GeometryBufferType type) const;

    NumRenderObjects _numObjects = {};
    ShadowGridLayout _shadowGridLayout;
    std::array<uint64_t, GeometryBufferType_Count> _allocationIds = {};
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
