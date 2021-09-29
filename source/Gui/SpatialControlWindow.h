#pragma once

#include "EngineImpl/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"

class _SpatialControlWindow
{
public:
    _SpatialControlWindow(
        SimulationController const& simController,
        Viewport const& viewport,
        StyleRepository const& styleRepository);

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    SimulationController _simController;
    Viewport _viewport;
    StyleRepository _styleRepository;

    bool _on = true;
};