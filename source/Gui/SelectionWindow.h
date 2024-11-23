#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"
#include "AlienWindow.h"
#include "Base/Singleton.h"

class SelectionWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SelectionWindow);

private:
    SelectionWindow();

    void processIntern();
};