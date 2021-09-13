#pragma once

#include "Definitions.h"

class _StyleRepository
{
public:
    _StyleRepository();

    ImFont* getLargeFont() const;

private:

    ImFont* _largeFont = nullptr;
};