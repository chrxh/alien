#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"
#include "EngineInterface/EngineConstants.h"

class _MassOperationsDialog
{
public:
    _MassOperationsDialog(SimulationController const& simController);

    void process();

    void show();

private:
    void colorCheckbox(std::string id, uint32_t cellColor, bool& check);

    void onExecute();
    bool isOkEnabled();
    void validationAndCorrection();

    SimulationController _simController;

    bool _show = false;

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

    bool _restrictToSelectedClusters = false;
};
