#pragma once

#include "WebBuilderFacade.h"

class WebBuilderFacadeImpl : public WebBuilderFacade
{
public:
    virtual ~WebBuilderFacadeImpl() = default;

    WebController* buildWebController() const override;
};

