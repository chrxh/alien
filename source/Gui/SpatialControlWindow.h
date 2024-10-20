#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class SpatialControlWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SpatialControlWindow);

private:
    SpatialControlWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processCenterButton();
    void processResizeButton();

    void processCenterOnSelection();

    SimulationFacade _simulationFacade;

    bool _centerSelection = false;
};