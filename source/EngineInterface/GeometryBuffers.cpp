#include "GeometryBuffers.h"

#include <glad/glad.h>

#include "Base/Definitions.h"

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
    glGenVertexArrays(1, &result->_vaoForConnectionArrows);
    glGenBuffers(1, &result->_vboForConnectionArrows);
    return GeometryBuffers(result);
}

void _GeometryBuffers::resizeIfNecessary(NumRenderObjects const& numRenderObjects)
{
    _numObjects = numRenderObjects;
    if (numRenderObjects.vertices >= _vertexBufferCapacity) {
        _vertexBufferCapacity = std::max(numRenderObjects.vertices * 2, static_cast<uint64_t>(100000));
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
        glBindBuffer(GL_ARRAY_BUFFER, getVboForConnectionArrows());
        glBufferData(GL_ARRAY_BUFFER, toInt(_connectionArrowVertexBufferCapacity * sizeof(ConnectionArrowVertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
}

NumRenderObjects _GeometryBuffers::getNumObjects() const
{
    return _numObjects;
}
