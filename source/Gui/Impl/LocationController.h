#pragma once

#include <Base/Interface/Definitions.h>
#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>

#include "Definitions.h"
#include "LocationWindow.h"
#include "MainLoopEntity.h"

class LocationController : public MainLoopEntity
{
    MAKE_SINGLETON(LocationController);

public:
    void addLocationWindow(int orderNumber, RealVector2D const& initialPos);

private:
    void init() override;
    void process() override;
    void shutdown() override {}

    std::vector<LocationWindow> _locationWindows;
    std::optional<int> _sessionId;
};
