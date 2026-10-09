#include "GeometryBuffers.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

uint64_t GeometryBufferLayout::getNumElements(NumRenderObjects const& numObjects, GeometryBufferType type)
{
    switch (type) {
    case GeometryBufferType_Objects:
        return numObjects.objects;
    case GeometryBufferType_FluidParticles:
        return numObjects.fluidParticles;
    case GeometryBufferType_Locations:
        return numObjects.locations;
    case GeometryBufferType_SelectedObjects:
        return numObjects.selectedObjects;
    case GeometryBufferType_LineIndices:
        return numObjects.lineIndices;
    case GeometryBufferType_TriangleIndices:
        return numObjects.triangleIndices;
    case GeometryBufferType_SelectedConnections:
        return numObjects.connectionArrowVertices;
    case GeometryBufferType_AttackEvents:
        return numObjects.attackEventVertices;
    case GeometryBufferType_DetonationEvents:
        return numObjects.detonationEventVertices;
    default:
        throw std::runtime_error("Invalid geometry buffer type.");
    }
}

void _GeometryBuffers::updateNumObjects(NumRenderObjects const& numRenderObjects)
{
    _numObjects = numRenderObjects;
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        auto numElements = GeometryBufferLayout::getNumElements(numRenderObjects, type);
        auto& capacity = _capacities.at(type);
        if (numElements >= capacity) {
            capacity = std::max(numElements * 2, GeometryBufferLayout::MinCapacities.at(type));
            reallocate(type, capacity * GeometryBufferLayout::ElementSizes.at(type));
            ++_allocationIds.at(type);
        }
    }
}

NumRenderObjects _GeometryBuffers::getNumObjects() const
{
    return _numObjects;
}

uint64_t _GeometryBuffers::getAllocationId(GeometryBufferType type) const
{
    return _allocationIds.at(type);
}

uint64_t _GeometryBuffers::getCapacity(GeometryBufferType type) const
{
    return _capacities.at(type);
}

template <typename T>
std::vector<T> _GeometryBuffers::downloadElements(GeometryBufferType type) const
{
    std::vector<T> result(GeometryBufferLayout::getNumElements(_numObjects, type));
    if (!result.empty()) {
        download(type, result.data(), result.size() * sizeof(T));
    }
    return result;
}

std::vector<ObjectVertexData> _GeometryBuffers::getCellData() const
{
    return downloadElements<ObjectVertexData>(GeometryBufferType_Objects);
}

std::vector<FluidParticleVertexData> _GeometryBuffers::getFluidParticleData() const
{
    return downloadElements<FluidParticleVertexData>(GeometryBufferType_FluidParticles);
}

std::vector<LocationVertexData> _GeometryBuffers::getLocationData() const
{
    return downloadElements<LocationVertexData>(GeometryBufferType_Locations);
}

std::vector<SelectedObjectVertexData> _GeometryBuffers::getSelectedObjectData() const
{
    return downloadElements<SelectedObjectVertexData>(GeometryBufferType_SelectedObjects);
}

std::vector<unsigned int> _GeometryBuffers::getLineIndices() const
{
    return downloadElements<unsigned int>(GeometryBufferType_LineIndices);
}

std::vector<unsigned int> _GeometryBuffers::getTriangleIndices() const
{
    return downloadElements<unsigned int>(GeometryBufferType_TriangleIndices);
}

std::vector<ConnectionArrowVertexData> _GeometryBuffers::getSelectedConnectionData() const
{
    return downloadElements<ConnectionArrowVertexData>(GeometryBufferType_SelectedConnections);
}

std::vector<AttackEventVertexData> _GeometryBuffers::getAttackEventData() const
{
    return downloadElements<AttackEventVertexData>(GeometryBufferType_AttackEvents);
}

std::vector<DetonationEventVertexData> _GeometryBuffers::getDetonationEventData() const
{
    return downloadElements<DetonationEventVertexData>(GeometryBufferType_DetonationEvents);
}

GeometryBuffers _HostGeometryBuffers::create()
{
    return GeometryBuffers(new _HostGeometryBuffers());
}

bool _HostGeometryBuffers::isMemoryShareable() const
{
    return false;
}

SharedGeometryMemory _HostGeometryBuffers::shareMemory(GeometryBufferType type)
{
    throw std::runtime_error("Host geometry buffers cannot be shared.");
}

void _HostGeometryBuffers::upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes)
{
    std::memcpy(_buffers.at(type).data(), data, sizeInBytes);
}

void _HostGeometryBuffers::download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const
{
    std::memcpy(data, _buffers.at(type).data(), sizeInBytes);
}

void _HostGeometryBuffers::reallocate(GeometryBufferType type, uint64_t sizeInBytes)
{
    _buffers.at(type).resize(sizeInBytes);
}
