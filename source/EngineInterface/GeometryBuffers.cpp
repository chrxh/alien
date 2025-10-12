#include "GeometryBuffers.h"

#include <glad/glad.h>

GeometryBuffers _GeometryBuffers::create()
{
    auto result = new _GeometryBuffers();
    glGenVertexArrays(1, &result->vao);
    glGenBuffers(1, &result->vbo);
    glGenBuffers(1, &result->ebo);
    return GeometryBuffers(result);
}
