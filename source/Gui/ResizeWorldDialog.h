#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _ResizeWorldDialog : public _AlienDialog
{
public:
    _ResizeWorldDialog(SimulationController const& simController);

    void open();

private:
    void processIntern() override;

    void onResizing();

    SimulationController _simController;

    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};
