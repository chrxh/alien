#pragma once

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _RandomizeDialog
{
public:
    _RandomizeDialog(SimulationController const& simController);

    void process();

    void show();

private:
    void colorCheckbox(std::string id, uint32_t cellColor, bool& check);

    void onRandomize();
    bool isOkEnabled();
    void validationAndCorrection();

    SimulationController _simController;

    bool _show = false;

    bool _randomizeColors = false;
    bool _checkColors[7] = {false, false, false, false, false, false, false};

    bool _randomizeEnergies = false;
    float _minEnergy = 200.0;
    float _maxEnergy = 200.0;

    bool _randomizeAges = false;
    int _minAge = 0;
    int _maxAge = 0;

    bool _restrictToSelectedClusters = false;
};
