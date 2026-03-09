#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/EngineConstants.h>

#include "AlienDialog.h"
#include "Definitions.h"

class MassOperationsDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(MassOperationsDialog);

private:
    MassOperationsDialog();

    void initIntern() override;
    void processIntern() override;

    void colorCheckbox(std::string id, uint32_t cellColor, bool& check);

    void onExecute();
    bool isOkEnabled();
    void validateAndCorrect();

    bool _randomizeCellColors = false;
    bool _checkedCellColors[MAX_COLORS] = {};

    bool _randomizeGenomeColors = false;
    bool _checkedGenomeColors[MAX_COLORS] = {};

    bool _randomizeEnergies = false;
    float _minEnergy = 200.0;
    float _maxEnergy = 200.0;

    bool _randomizeAges = false;
    int _minAge = 0;
    int _maxAge = 0;

    bool _randomizeCountdowns = false;
    int _minCountdown = 5;
    int _maxCountdown = 5;

    bool _randomizeLineageId = false;

    bool _randomizeGlow = false;
    float _minGlow = 0;
    float _maxGlow = 0.0f;

    bool _restrictToSelectedCreatures = true;
};
