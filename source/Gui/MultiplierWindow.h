#pragma once

#include "Base/Singleton.h"
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

class MultiplierWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MultiplierWindow);

private:
    MultiplierWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;
    void processGridPanel();
    void processRandomPanel();

    void validationAndCorrection();

    void onBuild();
    void onUndo();

    SimulationFacade _simulationFacade;

    MultiplierMode _mode = MultiplierMode_Grid;

    DescriptionEditService::GridMultiplyParameters _gridParameters;
    DescriptionEditService::RandomMultiplyParameters _randomParameters;

    DataDescription _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};