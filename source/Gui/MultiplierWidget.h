#pragma once

#include <Base/Singleton.h>

#include <Data/MultiplierService.h>

#include <EngineInterface/SelectionShallowData.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

using MultiplierMode = int;
enum MultiplierMode_
{
    MultiplierMode_Grid,
    MultiplierMode_Random
};

class MultiplierWidget : public MainLoopEntity
{
    MAKE_SINGLETON(MultiplierWidget);

public:
    void processContent();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void processGridPanel();
    void processRandomPanel();
    void processGridPreview() const;
    void validateAndCorrect();

    void onBuild();
    void onUndo();

    MultiplierMode _mode = MultiplierMode_Grid;
    MultiplierService::GridParameters _gridParameters;
    MultiplierService::RandomParameters _randomParameters;
    ContentDesc _origSelection;
    std::optional<SelectionShallowData> _selectionDataAfterMultiplication;
};
