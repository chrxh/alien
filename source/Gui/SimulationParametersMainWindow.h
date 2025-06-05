#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"

#include "AlienWindow.h"
#include "SimulationParametersBaseWidget.h"
#include "LayerColorPalette.h"

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

    struct Location
    {
        std::string name;
        LocationType type = LocationType::Layer;
        std::string position;
        std::string strength;
    };
    void processLocationTable();

    void processExpertSettings();

    void onOpenParameters();
    void onSaveParameters();
    void onInsertDefaultLayer();
    void onInsertDefaultSource();
    void onCloneLocation();
    void onDeleteLocation();
    void onDecreaseOrderNumber();
    void onIncreaseOrderNumber();
    void onOpenInLocationWindow();
    void onCenterLocation(int orderNumber);

    void updateLocations();

    void correctLayout(float origMasterHeight, float origExpertWidgetHeight);

    bool checkNumLayers(SimulationParameters const& parameters);
    bool checkNumSources(SimulationParameters const& parameters);

    float getMasterWidgetRefHeight() const;
    float getExpertWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;

private:
    SimulationFacade _simulationFacade;

    LocationWidget _baseWidgets;
    LocationWidget _layerWidgets;
    LocationWidget _sourceWidgets;

    bool _masterWidgetOpen = true;
    bool _detailWidgetOpen = true;
    bool _expertWidgetOpen = false;
    float _masterWidgetHeight = 0;
    float _expertWidgetHeight = 0;

    LayerColorPalette _layerColorPalette;

    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _sessionId;

    std::vector<Location> _locations;
    int _selectedOrderNumber = 0;

    int _locationWindowCounter = 0;
    int _insertedLocationCounter = 0;

    std::string _fileDialogPath;

    std::string _filter;
};
