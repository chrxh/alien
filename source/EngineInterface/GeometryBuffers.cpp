#include "GeometryBuffers.h"

#include <glad/glad.h>

#include <Base/Definitions.h>

GeometryBuffers _GeometryBuffers::create()
{
    auto result = new _GeometryBuffers();
    glGenVertexArrays(1, &result->_vaoForPointsAndLines);
    glGenBuffers(1, &result->_vboForCells);
    glGenBuffers(1, &result->_eboForLines);
    glGenVertexArrays(1, &result->_vaoForTriangles);
    glGenBuffers(1, &result->_eboForTriangles);
    glGenVertexArrays(1, &result->_vaoForEnergyParticles);
    glGenBuffers(1, &result->_vboForEnergyParticles);
    glGenVertexArrays(1, &result->_vaoForLocations);
    glGenBuffers(1, &result->_vboForLocations);
    glGenVertexArrays(1, &result->_vaoForSelectedObjects);
    glGenBuffers(1, &result->_vboForSelectedObjects);
    glGenVertexArrays(1, &result->_vaoForSelectedConnections);
    glGenBuffers(1, &result->_vboForSelectedConnections);
    glGenVertexArrays(1, &result->_vaoForAttackEvents);
    glGenBuffers(1, &result->_vboForAttackEvents);
    glGenVertexArrays(1, &result->_vaoForDetonationEvents);
    glGenBuffers(1, &result->_vboForDetonationEvents);
    return GeometryBuffers(result);
}

void _GeometryBuffers::updateNumObjects(NumRenderObjects const& numRenderObjects)
{
    _numObjects = numRenderObjects;
    if (numRenderObjects.cells >= _vertexBufferCapacity) {
        _vertexBufferCapacity = std::max(numRenderObjects.cells * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForCells());
        glBufferData(GL_ARRAY_BUFFER, toInt(_vertexBufferCapacity * sizeof(CellVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.energyParticles >= _energyParticleBufferCapacity) {
        _energyParticleBufferCapacity = std::max(numRenderObjects.energyParticles * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForEnergyParticles());
        glBufferData(GL_ARRAY_BUFFER, toInt(_energyParticleBufferCapacity * sizeof(EnergyParticleVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.locations >= _locationBufferCapacity) {
        _locationBufferCapacity = std::max(numRenderObjects.locations * 2, static_cast<uint64_t>(1000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForLocations());
        glBufferData(GL_ARRAY_BUFFER, toInt(_locationBufferCapacity * sizeof(LocationVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.selectedObjects >= _selectedObjectBufferCapacity) {
        _selectedObjectBufferCapacity = std::max(numRenderObjects.selectedObjects * 2, static_cast<uint64_t>(10000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedObjects());
        glBufferData(GL_ARRAY_BUFFER, toInt(_selectedObjectBufferCapacity * sizeof(SelectedObjectVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.lineIndices >= _lineIndexBufferCapacity) {
        _lineIndexBufferCapacity = std::max(numRenderObjects.lineIndices * 2, static_cast<uint64_t>(100000));
        glBindVertexArray(getVaoForPointsAndLines());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForLines());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(_lineIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.triangleIndices >= _triangleIndexBufferCapacity) {
        _triangleIndexBufferCapacity = std::max(numRenderObjects.triangleIndices * 2, static_cast<uint64_t>(100000));
        glBindVertexArray(getVaoForTriangles());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForTriangles());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(_triangleIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.connectionArrowVertices >= _connectionArrowVertexBufferCapacity) {
        _connectionArrowVertexBufferCapacity = std::max(numRenderObjects.connectionArrowVertices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedConnections());
        glBufferData(GL_ARRAY_BUFFER, toInt(_connectionArrowVertexBufferCapacity * sizeof(ConnectionArrowVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.attackEventVertices >= _attackEventVertexBufferCapacity) {
        _attackEventVertexBufferCapacity = std::max(numRenderObjects.attackEventVertices * 2, static_cast<uint64_t>(10000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForAttackEvents());
        glBufferData(GL_ARRAY_BUFFER, toInt(_attackEventVertexBufferCapacity * sizeof(AttackEventVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.detonationEventVertices >= _detonationEventVertexBufferCapacity) {
        _detonationEventVertexBufferCapacity = std::max(numRenderObjects.detonationEventVertices * 2, static_cast<uint64_t>(10000));
        glBindBuffer(GL_ARRAY_BUFFER, getVboForDetonationEvents());
        glBufferData(GL_ARRAY_BUFFER, toInt(_detonationEventVertexBufferCapacity * sizeof(DetonationEventVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
}

NumRenderObjects _GeometryBuffers::getNumObjects() const
{
    return _numObjects;
}

void _GeometryBuffers::uploadCellData(CellVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForCells());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(CellVertexData)), data);
}

void _GeometryBuffers::uploadEnergyParticleData(EnergyParticleVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForEnergyParticles());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(EnergyParticleVertexData)), data);
}

void _GeometryBuffers::uploadLocationData(LocationVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForLocations());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(LocationVertexData)), data);
}

void _GeometryBuffers::uploadSelectedObjectData(SelectedObjectVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedObjects());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(SelectedObjectVertexData)), data);
}

void _GeometryBuffers::uploadLineIndices(unsigned int const* data, uint64_t count)
{
    if (count == 0) return;
    glBindVertexArray(getVaoForPointsAndLines());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForLines());
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(count * sizeof(unsigned int)), data);
}

void _GeometryBuffers::uploadTriangleIndices(unsigned int const* data, uint64_t count)
{
    if (count == 0) return;
    glBindVertexArray(getVaoForTriangles());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForTriangles());
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(count * sizeof(unsigned int)), data);
}

void _GeometryBuffers::uploadSelectedConnectionData(ConnectionArrowVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedConnections());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(ConnectionArrowVertexData)), data);
}

void _GeometryBuffers::uploadAttackEventData(AttackEventVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForAttackEvents());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(AttackEventVertexData)), data);
}

void _GeometryBuffers::uploadDetonationEventData(DetonationEventVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForDetonationEvents());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(DetonationEventVertexData)), data);
}

std::vector<CellVertexData> _GeometryBuffers::downloadCellData() const
{
    std::vector<CellVertexData> result(_numObjects.cells);
    if (_numObjects.cells == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForCells);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.cells * sizeof(CellVertexData)), result.data());
    return result;
}

std::vector<EnergyParticleVertexData> _GeometryBuffers::downloadEnergyParticleData() const
{
    std::vector<EnergyParticleVertexData> result(_numObjects.energyParticles);
    if (_numObjects.energyParticles == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForEnergyParticles);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.energyParticles * sizeof(EnergyParticleVertexData)), result.data());
    return result;
}

std::vector<LocationVertexData> _GeometryBuffers::downloadLocationData() const
{
    std::vector<LocationVertexData> result(_numObjects.locations);
    if (_numObjects.locations == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForLocations);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.locations * sizeof(LocationVertexData)), result.data());
    return result;
}

std::vector<SelectedObjectVertexData> _GeometryBuffers::downloadSelectedObjectData() const
{
    std::vector<SelectedObjectVertexData> result(_numObjects.selectedObjects);
    if (_numObjects.selectedObjects == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForSelectedObjects);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.selectedObjects * sizeof(SelectedObjectVertexData)), result.data());
    return result;
}

std::vector<unsigned int> _GeometryBuffers::downloadLineIndices() const
{
    std::vector<unsigned int> result(_numObjects.lineIndices);
    if (_numObjects.lineIndices == 0) return result;
    glBindVertexArray(_vaoForPointsAndLines);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eboForLines);
    glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(_numObjects.lineIndices * sizeof(unsigned int)), result.data());
    return result;
}

std::vector<unsigned int> _GeometryBuffers::downloadTriangleIndices() const
{
    std::vector<unsigned int> result(_numObjects.triangleIndices);
    if (_numObjects.triangleIndices == 0) return result;
    glBindVertexArray(_vaoForTriangles);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eboForTriangles);
    glGetBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(_numObjects.triangleIndices * sizeof(unsigned int)), result.data());
    return result;
}

std::vector<ConnectionArrowVertexData> _GeometryBuffers::downloadSelectedConnectionData() const
{
    std::vector<ConnectionArrowVertexData> result(_numObjects.connectionArrowVertices);
    if (_numObjects.connectionArrowVertices == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForSelectedConnections);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.connectionArrowVertices * sizeof(ConnectionArrowVertexData)), result.data());
    return result;
}

std::vector<AttackEventVertexData> _GeometryBuffers::downloadAttackEventData() const
{
    std::vector<AttackEventVertexData> result(_numObjects.attackEventVertices);
    if (_numObjects.attackEventVertices == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForAttackEvents);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.attackEventVertices * sizeof(AttackEventVertexData)), result.data());
    return result;
}

std::vector<DetonationEventVertexData> _GeometryBuffers::downloadDetonationEventData() const
{
    std::vector<DetonationEventVertexData> result(_numObjects.detonationEventVertices);
    if (_numObjects.detonationEventVertices == 0) return result;
    glBindBuffer(GL_ARRAY_BUFFER, _vboForDetonationEvents);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, toInt(_numObjects.detonationEventVertices * sizeof(DetonationEventVertexData)), result.data());
    return result;
}
