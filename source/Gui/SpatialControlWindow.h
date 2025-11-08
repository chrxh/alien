#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>

#include "AlienWindow.h"
#include "Definitions.h"

class SpatialControlWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SpatialControlWindow);

private:
    SpatialControlWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processCenterButton();
    void processResizeButton();

    void processCenterOnSelection();

    bool _centerSelection = false;
};
