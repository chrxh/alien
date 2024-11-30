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
    void processExpertModes();
    void processStatusBar();

    enum class LocationType
    {
        Base,
        ParameterZone,
        RadiationSource
    };
    struct Location
    {
        std::string name;
        LocationType type = LocationType::ParameterZone;
        std::string position;
        std::string strength;
    };
    void processLocationTable();

    std::vector<Location> generateLocations() const;

    void correctLayout(float origMasterHeight, float origExpertWidgetHeight);

    float getMasterWidgetRefHeight() const;
    float getExpertWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;

private:
    SimulationFacade _simulationFacade;

    bool _masterOpen = true;
    bool _detailOpen = true;
    bool _expertModesOpen = true;
    float _masterHeight = 0;
    float _expertWidgetHeight = 0;

    std::optional<SimulationParameters> _copiedParameters;
};