#pragma once

#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class _RadiationSourcesWindow : public AlienWindow
{
public:
    _RadiationSourcesWindow(SimulationFacade const& simulationFacade);

private:
    void processIntern() override;

    bool processTab(int index); //returns false if tab should be closed
    void onAppendTab();
    void onDeleteTab(int index);

    RadiationSource createParticleSource() const;

    void validationAndCorrection(RadiationSource& source) const;

    SimulationFacade _simulationFacade;
};