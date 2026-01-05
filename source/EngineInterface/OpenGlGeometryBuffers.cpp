#include "OpenGlGeometryBuffers.h"

#include <glad/glad.h>

#include <Base/Definitions.h>

GeometryBuffers _OpenGlGeometryBuffers::create()
{
    auto result = new _OpenGlGeometryBuffers();
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

void _OpenGlGeometryBuffers::updateNumObjects(NumRenderObjects const& numRenderObjects)
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

NumRenderObjects _OpenGlGeometryBuffers::getNumObjects() const
{
    return _numObjects;
}

void _OpenGlGeometryBuffers::uploadCellData(CellVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForCells());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(CellVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadEnergyParticleData(EnergyParticleVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForEnergyParticles());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(EnergyParticleVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadLocationData(LocationVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForLocations());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(LocationVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadSelectedObjectData(SelectedObjectVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedObjects());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(SelectedObjectVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadLineIndices(unsigned int const* data, uint64_t count)
{
    if (count == 0) return;
    glBindVertexArray(getVaoForPointsAndLines());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForLines());
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(count * sizeof(unsigned int)), data);
}

void _OpenGlGeometryBuffers::uploadTriangleIndices(unsigned int const* data, uint64_t count)
{
    if (count == 0) return;
    glBindVertexArray(getVaoForTriangles());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEboForTriangles());
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, toInt(count * sizeof(unsigned int)), data);
}

void _OpenGlGeometryBuffers::uploadSelectedConnectionData(ConnectionArrowVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForSelectedConnections());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(ConnectionArrowVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadAttackEventData(AttackEventVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForAttackEvents());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(AttackEventVertexData)), data);
}

void _OpenGlGeometryBuffers::uploadDetonationEventData(DetonationEventVertexData const* data, uint64_t count)
{
    if (count == 0) return;
    glBindBuffer(GL_ARRAY_BUFFER, getVboForDetonationEvents());
    glBufferSubData(GL_ARRAY_BUFFER, 0, toInt(count * sizeof(DetonationEventVertexData)), data);
}
