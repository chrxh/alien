#include "GeometrySource.h"

#include <glad/glad.h>

GeometrySource _GeometrySource::create()
{
    auto result = new _GeometrySource();
    glGenVertexArrays(1, &result->vao);
    glGenBuffers(1, &result->vbo);
    glGenBuffers(1, &result->ebo);
    return GeometrySource(result);
}
