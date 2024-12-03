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

    void onOpenParameters();
    void onSaveParameters();
    void onAddZone();
    void onAddSource();
    void onCloneLocation();
    void onDecreaseLocationIndex();
    void onIncreaseLocationIndex();

    void updateLocations();

    void setDefaultShapeDataForZone(SimulationParametersZone& spot) const;

    void correctLayout(float origMasterHeight, float origExpertWidgetHeight);

    float getMasterWidgetRefHeight() const;
    float getExpertWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;

private:
    SimulationFacade _simulationFacade;

    bool _masterWidgetOpen = true;
    bool _detailWidgetOpen = true;
    bool _expertWidgetOpen = true;
    float _masterWidgetHeight = 0;
    float _expertWidgetHeight = 0;

    uint32_t _savedPalette[32] = {};

    std::optional<SimulationParameters> _copiedParameters;

    std::vector<Location> _locations;
    std::optional<int> _selectedLocationIndex;

    std::string _fileDialogPath;
};