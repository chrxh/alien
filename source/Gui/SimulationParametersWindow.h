#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpot.h"
#include "EngineInterface/SimulationParameters.h"
#include "Definitions.h"
#include "AlienWindow.h"

class _SimulationParametersWindow : public _AlienWindow
{
public:
    _SimulationParametersWindow(SimulationController const& simController, RadiationSourcesWindow const& radiationSourcesWindow);
    ~_SimulationParametersWindow();

private:
    void processIntern() override;

    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);
    void createDefaultSpotData(SimulationParametersSpot& spot);

    void processToolbar();
    void processTabWidget(SimulationParameters& parameters, SimulationParameters const& lastParameters, SimulationParameters& origParameters);
    void processBase(SimulationParameters& parameters, SimulationParameters const& origParameters);
    void processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot, SimulationParameters const& parameters);
    void processFeatureList(SimulationParameters& parameters, SimulationParameters const& lastParameters, SimulationParameters const& origParameters);

    void onOpenParameters();
    void onSaveParameters();

    void validationAndCorrectionLayout();
    void validationAndCorrection(SimulationParameters& parameters) const;
    void validationAndCorrection(SimulationParametersSpot& spot, SimulationParameters const& parameters) const;

    SimulationController _simController;
    RadiationSourcesWindow _radiationSourcesWindow;

    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
    std::string _startingPath;
    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _numSpotsLastTime;
    std::vector<std::string> _cellFunctionStrings;

    bool _featureListOpen = false;
    float _featureListHeight = 200.0f;
};