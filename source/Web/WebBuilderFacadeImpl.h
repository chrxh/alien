#pragma once

#include "WebBuilderFacade.h"

class WebBuilderFacadeImpl : public WebBuilderFacade
{
public:
    virtual ~WebBuilderFacadeImpl() = default;

    WebAccess* buildWebController() const override;
};

