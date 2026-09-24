#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/SimulationParameters.h>

#include "AlienWindow.h"
#include "SimulationParametersBaseWidget.h"

class SimulationParametersMainWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationParametersMainWindow);

private:
    SimulationParametersMainWindow();

    void initIntern() override;
    void processIntern() override;
    void shutdownIntern() override;

    void processToolbar();
    void processMasterWidget();
    void processDetailWidget();
    void processExpertWidget();
    void processStatusBar();

    void startFilterTypingIfNeeded();

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

    void showMaxLocationsReachedMessage(LocationType locationType) const;

    float getMasterWidgetRefHeight() const;
    float getExpertWidgetRefHeight() const;

    float getMasterWidgetHeight() const;
    float getDetailWidgetHeight() const;

private:

    LocationWidget _baseWidgets;
    LocationWidget _layerWidgets;
    LocationWidget _sourceWidgets;

    bool _masterWidgetOpen = true;
    bool _detailWidgetOpen = true;
    bool _expertWidgetOpen = false;
    float _masterWidgetHeight = 0;
    float _expertWidgetHeight = 0;

    std::optional<SimulationParameters> _copiedParameters;
    std::optional<int> _sessionId;

    std::vector<Location> _locations;
    int _selectedOrderNumber = 0;
    int _selectedLocationId = 0;

    int _locationWindowCounter = 0;

    std::string _fileDialogPath;

    std::string _filter;
    std::vector<unsigned int> _pendingFilterChars;
};
