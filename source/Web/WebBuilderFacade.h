#pragma once

#include "Definitions.h"

class WebBuilderFacade
{
public:
    virtual ~WebBuilderFacade() = default;

    virtual WebAccess* buildWebController() const = 0;
};

