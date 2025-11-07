#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>

#include "AlienWindow.h"
#include "Definitions.h"

class SelectionWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SelectionWindow);

private:
    SelectionWindow();

    void processIntern() override;
    bool isShown() override;
};
