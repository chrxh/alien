#pragma once

#include "Definitions.h"

struct UnprocessedTask {
    string id;
    IntVector2D pos;
    IntVector2D size;
};

struct ProcessedTask {
    string id;
    IntVector2D pos;
    IntVector2D size;
    QByteArray data;
};
