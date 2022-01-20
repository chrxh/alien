#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SelectionShallowData.h"

#include "Definitions.h"
#include "AlienWindow.h"

enum class MultiplierMode
{
    Grid,
    Random
};

class _MultiplierWindow : public _AlienWindow
{
public:
    _MultiplierWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport);

private:
    void processIntern() override;
    void processGridPanel();
    void processRandomPanel();

    EditorModel _editorModel; 
    SimulationController _simController;
    Viewport _viewport;

    MultiplierMode _mode = MultiplierMode::Grid;

    DescriptionHelper::GridMultiplyParameters _gridParameters;
    DescriptionHelper::RandomMultiplyParameters _randomParameters;

    DataDescription _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};