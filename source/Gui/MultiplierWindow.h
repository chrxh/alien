#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/MultiplierService.h>
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
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processGridPanel();
    void processRandomPanel();

    void validateAndCorrect();

    void onBuild();
    void onUndo();

    MultiplierMode _mode = MultiplierMode_Grid;

    MultiplierService::GridParameters _gridParameters;
    MultiplierService::RandomParameters _randomParameters;

    ContentDesc _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};
