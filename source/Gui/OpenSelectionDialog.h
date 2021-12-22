#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _OpenSelectionDialog
{
public:
    _OpenSelectionDialog(
        EditorModel const& editorModel,
        SimulationController const& simController,
        Viewport const& viewport);

    void process();

    void show();

private:
    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
};
