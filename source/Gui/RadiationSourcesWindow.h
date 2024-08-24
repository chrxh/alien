#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class _RadiationSourcesWindow : public _AlienWindow
{
public:
    _RadiationSourcesWindow(SimulationController const& simController);

private:
    void processIntern() override;

    bool processTab(int index); //return false if tab is closed
    void processAppendTab();
    void processDelTab(int index);

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationController _simController;
};