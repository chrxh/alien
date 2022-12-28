#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class _ParticleSourcesWindow : public _AlienWindow
{
public:
    _ParticleSourcesWindow(SimulationController const& simController);

private:
    void processIntern() override;

    ParticleSource createParticleSource() const;

    SimulationController _simController;
};