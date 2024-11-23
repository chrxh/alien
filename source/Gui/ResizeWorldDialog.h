#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ResizeWorldDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ResizeWorldDialog);

public:
    void open() override;

private:
    ResizeWorldDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;

    void onResizing();

    SimulationFacade _simulationFacade;

    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};
