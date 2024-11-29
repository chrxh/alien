#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"

#include "AlienWindow.h"

class SimulationParametersWindowPrototype : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationParametersWindowPrototype);

private:
    SimulationParametersWindowPrototype();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;
    void shutdownIntern() override;

    void processToolbar();
    void processMasterEditor();
    void processDetailEditor();
    void processAddonList();
    void processStatusBar();

    void processRegionTable();

    void correctLayout();

    float getMasterWidgetRefHeight() const;
    float getAddonWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;
    float getAddonWidgetHeight() const;

private:
    SimulationFacade _simulationFacade;

    bool _masterOpen = true;
    bool _detailOpen = true;
    bool _addonOpen = true;
    float _masterHeight = 0;
    float _addonHeight = 0;

    std::optional<SimulationParameters> _copiedParameters;

    struct Region{};
    std::vector<Region> _regions;
};