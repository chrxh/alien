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
    bool _checkedCellColors[MAX_COLORS] = {false, false, false, false, false, false, false};

    bool _randomizeGenomeColors = false;
    bool _checkedGenomeColors[MAX_COLORS] = {false, false, false, false, false, false, false};

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

    bool _setGlow = false;
    float _glowValue = 0;

    bool _restrictToSelectedCreatures = true;
};
