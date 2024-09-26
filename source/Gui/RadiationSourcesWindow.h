#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class _RadiationSourcesWindow : public _AlienWindow
{
public:
    _RadiationSourcesWindow(SimulationController const& simController, SimulationInteractionController const& simInteractionController);

private:
    void processIntern() override;

    bool processTab(int index); //returns false if tab should be closed
    void onAppendTab();
    void onDeleteTab(int index);

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationController _simController;
    SimulationInteractionController _simInteractionController;
};