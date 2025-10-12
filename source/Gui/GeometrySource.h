#pragma once

#include "Definitions.h"

class _GeometrySource
{
public:
    static GeometrySource create();

    unsigned int getVao() const { return vao; }
    unsigned int getVbo() const { return vbo; }
    unsigned int getEbo() const { return ebo; }

private:
    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int ebo = 0;

    _GeometrySource() = default;
};
