#pragma once

#include "Base/Singleton.h"

#include "AlienDialog.h"
#include "Definitions.h"

class ResizeWorldDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(ResizeWorldDialog);

public:
    void init(SimulationFacade const& simulationFacade);

    void open();

private:
    ResizeWorldDialog();

    void processIntern() override;

    void onResizing();

    SimulationFacade _simulationFacade;

    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};
