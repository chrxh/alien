#pragma once

#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>

#include <Engine/Interface/Definitions.h>

#include "AlienWindow.h"
#include "Definitions.h"

class SpatialControlWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SpatialControlWindow);

private:
    SpatialControlWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void processToolbar();

    void processCenterOnSelection();

    bool _centerSelection = false;
};
