#pragma once

#include "Base/Singleton.h"

#include "AlienWindow.h"
#include "Definitions.h"

class ShaderWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ShaderWindow);

private:
    ShaderWindow();

    void processIntern() override;
};