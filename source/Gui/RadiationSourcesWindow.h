#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class _RadiationSourcesWindow : public _AlienWindow
{
public:
    _RadiationSourcesWindow(SimulationController const& simController);

private:
    void processIntern() override;

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationController _simController;
};