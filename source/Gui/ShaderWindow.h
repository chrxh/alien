#pragma once

#include "AlienWindow.h"
#include "Definitions.h"

class _ShaderWindow : public AlienWindow
{
public:
    _ShaderWindow();
    ~_ShaderWindow() override;

private:
    void processIntern() override;
};