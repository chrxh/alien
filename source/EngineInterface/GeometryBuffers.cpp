#include "GeometryBuffers.h"

#include <glad/glad.h>

#include "Base/Definitions.h"

GeometryBuffers _GeometryBuffers::create()
{
    auto result = new _GeometryBuffers();
    glGenVertexArrays(1, &result->vao);
    glGenBuffers(1, &result->vbo);
    glGenBuffers(1, &result->ebo);
    return GeometryBuffers(result);
}

void _GeometryBuffers::resizeIfNecessary(NumRenderObjects const& numRenderObjects)
{
    if (numRenderObjects.vertices >= vertexBufferCapacity) {
        vertexBufferCapacity = std::max(numRenderObjects.vertices * 2, static_cast<uint64_t>(100000));
        glBindBuffer(GL_ARRAY_BUFFER, getVbo());
        glBufferData(GL_ARRAY_BUFFER, toInt(vertexBufferCapacity * sizeof(VertexData)), nullptr, GL_DYNAMIC_DRAW);
    }
    if (numRenderObjects.lineIndices >= lineIndexBufferCapacity) {
        lineIndexBufferCapacity = std::max(numRenderObjects.lineIndices * 2, static_cast<uint64_t>(100000));
        glBindVertexArray(getVao());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, getEbo());
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, toInt(lineIndexBufferCapacity * sizeof(unsigned int)), nullptr, GL_DYNAMIC_DRAW);
    }
}
