#include "GeometryBuffers.h"

#include <glad/glad.h>

#include "Base/Definitions.h"

GeometryBuffers _GeometryBuffers::create()
{
    auto result = new _GeometryBuffers();
    glGenVertexArrays(1, &result->_vao);
    glGenBuffers(1, &result->_vbo);
    glGenBuffers(1, &result->_ebo);
    glGenBuffers(1, &result->_tbo);
    return GeometryBuffers(result);
}

void _GeometryBuffers::resizeIfNecessary(NumRenderObjects const& numRenderObjects)
{
    if (numRenderObjects.vertices >= _vertexBufferCapacity) {
        _vertexBufferCapacity = std::max(numRenderObjects.vertices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, getVbo());
        glBufferData(GL_ARRAY_BUFFER, toInt(_vertexBufferCapacity * sizeof(VertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.lineIndices >= _lineIndexBufferCapacity) {
        _lineIndexBufferCapacity = std::max(numRenderObjects.lineIndices * 2, static_cast<uint64_t>(100000));
        glBindVertexArray(getVao());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEbo());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(_lineIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.triangleIndices >= _triangleIndexBufferCapacity) {
        _triangleIndexBufferCapacity = std::max(numRenderObjects.triangleIndices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getTbo());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(_triangleIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
}
