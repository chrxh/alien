#pragma once

#include "Base/Definitions.h"
#include "Base/Singleton.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "LocationWindow.h"
#include "MainLoopEntity.h"

class LocationController : public MainLoopEntity<SimulationFacade>
{
    MAKE_SINGLETON(LocationController);

public:
    void addLocationWindow(int orderNumber, RealVector2D const& initialPos);
    void deleteLocationWindow(int orderNumber);
    void remapLocationIndices(std::map<int, int> const& newByOldOrderNumber);

private:
    void init(SimulationFacade simulationFacade) override;
    void process() override;
    void shutdown() override {}

    SimulationFacade _simulationFacade;

    std::vector<LocationWindow> _locationWindows;
    std::optional<int> _sessionId;
};
