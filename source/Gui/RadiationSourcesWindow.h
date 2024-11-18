#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "AlienWindow.h"

class RadiationSourcesWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(RadiationSourcesWindow);

private:
    RadiationSourcesWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;

    void processBaseTab();
    bool processSourceTab(int index); //returns false if tab should be closed

    void onAppendTab();
    void onDeleteTab(int index);

    RadiationSource createParticleSource() const;

    void validateAndCorrect(RadiationSource& source) const;

    struct StrengthRatios
    {
        std::vector<float> values;
        std::set<int> pinned;
    };
    StrengthRatios getStrengthRatios(SimulationParameters const& parameters) const;
    void applyStrengthRatios(SimulationParameters& parameters, StrengthRatios const& ratios);

    void adaptStrengthRatios(StrengthRatios& ratios, StrengthRatios& origRatios) const;
    StrengthRatios calcStrengthRatiosForAddingSpot(StrengthRatios const& ratios) const;

    SimulationFacade _simulationFacade;

    std::optional<int> _sessionId;
    bool _focusBaseTab = false;
};