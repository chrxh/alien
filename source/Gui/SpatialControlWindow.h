#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class SpatialControlWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SpatialControlWindow);

public:
    void init(SimulationFacade const& simulationFacade);

private:
    SpatialControlWindow();

    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processCenterButton();
    void processResizeButton();

    void processCenterOnSelection();

    SimulationFacade _simulationFacade;
    ResizeWorldDialog _resizeWorldDialog;

    bool _centerSelection = false;
};