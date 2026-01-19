#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SelectionShallowData.h>

#include "AlienWindow.h"
#include "Definitions.h"

using MultiplierMode = int;
enum MultiplierMode_
{
    MultiplierMode_Grid,
    MultiplierMode_Random
};

class MultiplierWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MultiplierWindow);

private:
    MultiplierWindow();

    void initIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processGridPanel();
    void processRandomPanel();

    void validateAndCorrect();

    void onBuild();
    void onUndo();

    MultiplierMode _mode = MultiplierMode_Grid;

    DescriptionEditService::GridMultiplyParameters _gridParameters;
    DescriptionEditService::RandomMultiplyParameters _randomParameters;

    Desc _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};
