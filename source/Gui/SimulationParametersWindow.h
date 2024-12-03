#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersZone.h"
#include "EngineInterface/SimulationParameters.h"
#include "Definitions.h"
#include "AlienWindow.h"

class SimulationParametersWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationParametersWindow);

private:
    SimulationParametersWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;

    SimulationParametersZone createSpot(SimulationParameters const& simParameters, int index);
    void createDefaultSpotData(SimulationParametersZone& spot);

    void processToolbar();
    void processTabWidget();
    void processBase();
    bool processSpot(int index);    //returns false if tab should be closed
    void processAddonList();
    void processStatusBar();

    void onAppendTab();
    void onDeleteTab(int index);

    void onOpenParameters();
    void onSaveParameters();

    void validateAndCorrectLayout();

    SimulationFacade _simulationFacade;

    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor = 0;
    std::string _startingPath;
    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _sessionId;
    bool _focusBaseTab = false;
    std::vector<std::string> _cellFunctionStrings;

    bool _featureListOpen = false;
    float _featureListHeight = 200.0f;
};