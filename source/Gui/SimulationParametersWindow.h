#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpot.h"
#include "EngineInterface/SimulationParameters.h"
#include "Definitions.h"
#include "AlienWindow.h"

class _SimulationParametersWindow : public _AlienWindow
{
public:
    _SimulationParametersWindow(
        SimulationController const& simController,
        RadiationSourcesWindow const& radiationSourcesWindow,
        ModeController const& modeController);
    ~_SimulationParametersWindow();

private:
    void processIntern() override;

    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);
    void createDefaultSpotData(SimulationParametersSpot& spot);

    void processToolbar();
    void processTabWidget();
    void processBase();
    bool processSpot(int index);    //returns false if tab should be closed
    void processAddonList();

    void onAppendTab();
    void onDeleteTab(int index);

    void onOpenParameters();
    void onSaveParameters();

    void validationAndCorrectionLayout();
    void validationAndCorrection(SimulationParameters& parameters) const;
    void validationAndCorrection(SimulationParametersSpot& spot, SimulationParameters const& parameters) const;

    SimulationController _simController;
    ModeController _modeController;
    RadiationSourcesWindow _radiationSourcesWindow;

    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
    std::string _startingPath;
    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _sessionId;
    bool _focusBaseTab = false;
    std::vector<std::string> _cellFunctionStrings;

    bool _featureListOpen = false;
    float _featureListHeight = 200.0f;

    std::function<bool(void)> _getMousePickerEnabledFunc;
    std::function<void(bool)> _setMousePickerEnabledFunc;
    std::function<std::optional<RealVector2D>(void)> _getMousePickerPositionFunc;
};