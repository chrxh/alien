#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SelectionShallowData.h"

#include "Definitions.h"
#include "AlienWindow.h"

using MultiplierMode = int;
enum MultiplierMode_
{
    MultiplierMode_Grid,
    MultiplierMode_Random
};

class _MultiplierWindow : public _AlienWindow
{
public:
    _MultiplierWindow(EditorModel const& editorModel, SimulationController const& simController);

private:
    void processIntern() override;
    void processGridPanel();
    void processRandomPanel();

    void validationAndCorrection();

    void onBuild();
    void onUndo();

    EditorModel _editorModel; 
    SimulationController _simController;

    MultiplierMode _mode = MultiplierMode_Grid;

    DescriptionEditService::GridMultiplyParameters _gridParameters;
    DescriptionEditService::RandomMultiplyParameters _randomParameters;

    DataDescription _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};