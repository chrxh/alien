#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include <Base/Math.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& universeSize)
{
    _simulationFacade = _SimulationFacade::get();
    SimulationParameters parameters;
    for (int i = 0; i < MAX_COLORS; ++i) {
        parameters.radiationType1_strength.baseValue[i] = 0;
    }
    _simulationFacade->newSimulation(0, universeSize, parameters);
    _parameters = _simulationFacade->getSimulationParameters();
}

IntegrationTestFramework::~IntegrationTestFramework()
{
    _simulationFacade->closeSimulation();
}

double IntegrationTestFramework::getEnergy(Description const& data) const
{
    double result = 0;
    for (auto const& cell : data._cells) {
        result += cell._energy;
    }
    for (auto const& creature : data._creatures) {
        for (auto const& cell : creature._cells) {
            result += cell._energy;
        }
    }
    for (auto const& particle : data._particles) {
        result += particle._energy;
    }
    return result;
}

bool IntegrationTestFramework::compare(Description left, Description right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(CellDescription left, CellDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(ParticleDescription left, ParticleDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}
