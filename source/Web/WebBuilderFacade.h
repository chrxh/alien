#pragma once

#include "Definitions.h"

class WebBuilderFacade
{
public:
    virtual ~WebBuilderFacade() = default;

    virtual WebController* buildWebController() const = 0;
};

