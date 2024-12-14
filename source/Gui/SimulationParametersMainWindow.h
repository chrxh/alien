#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"

#include "AlienWindow.h"
#include "SimulationParametersBaseWidgets.h"
#include "ZoneColorPalette.h"

class SimulationParametersMainWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationParametersMainWindow);

private:
    SimulationParametersMainWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;
    void shutdownIntern() override;

    void processToolbar();
    void processMasterWidget();
    void processDetailWidget();
    void processExpertWidget();
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

    void processExpertSettings();

    void onOpenParameters();
    void onSaveParameters();
    void onAddZone();
    void onAddSource();
    void onCloneLocation();
    void onDeleteLocation();
    void onDecreaseLocationIndex();
    void onIncreaseLocationIndex();
    void onOpenInLocationWindow();
    void onCenterLocation(int locationIndex);

    void updateLocations();

    void setDefaultShapeDataForZone(SimulationParametersZone& spot) const;

    void correctLayout(float origMasterHeight, float origExpertWidgetHeight);

    bool checkNumZones(SimulationParameters const& parameters);
    bool checkNumSources(SimulationParameters const& parameters);

    float getMasterWidgetRefHeight() const;
    float getExpertWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;

private:
    SimulationFacade _simulationFacade;

    LocationWidgets _baseWidgets;
    LocationWidgets _zoneWidgets;
    LocationWidgets _sourceWidgets;

    bool _masterWidgetOpen = true;
    bool _detailWidgetOpen = true;
    bool _expertWidgetOpen = false;
    float _masterWidgetHeight = 0;
    float _expertWidgetHeight = 0;

    ZoneColorPalette _zoneColorPalette;

    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _sessionId;

    std::vector<Location> _locations;
    int _selectedLocationIndex = 0;

    int _locationWindowCounter = 0;

    std::string _fileDialogPath;
};
