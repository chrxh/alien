#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include <EngineImpl/SimulationFacadeImpl.h>

IntegrationTestFramework::TestSuiteContext IntegrationTestFramework::_globalContext;

void IntegrationTestFramework::TestSuiteContext::cleanup()
{
    if (simulationFacade) {
        simulationFacade->closeSimulation();
        simulationFacade.reset();
    }
}

void IntegrationTestFramework::cleanupGlobalContext()
{
    _globalContext.cleanup();
}

IntegrationTestFramework::IntegrationTestFramework(IntVector2D const& worldSize)
    : _worldSize(worldSize)
{
    if (_globalContext.simulationFacade == nullptr) {
        _globalContext.simulationFacade = std::make_shared<_SimulationFacadeImpl>();
    }

    _simulationFacade = _globalContext.simulationFacade;
    if (_simulationFacade->getWorldSize() != worldSize) {
        _simulationFacade->closeSimulation();
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->newSimulation(0, _worldSize, _parameters);
    } else {
        _simulationFacade->clear();
        _simulationFacade->setPreviewData(Description());
        _simulationFacade->setCurrentTimestepForPreview(0);
        _simulationFacade->setCurrentTimestep(0);
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

}

IntegrationTestFramework::~IntegrationTestFramework()
{
}

double IntegrationTestFramework::getEnergy(Description const& data) const
{
    auto getDepotEnergy = [](ObjectDescription const& object) -> double {
        if (object.getCellType() == CellType_Depot) {
            auto const& depot = std::get<DepotDescription>(object._cellType);
            return depot._storedUsableEnergy;
        }
        return 0;
    };

    double result = 0;
    for (auto const& object : data._objects) {
        result += object._usableEnergy + object._rawEnergy + getDepotEnergy(object);
    }
    for (auto const& energyParticle : data._energies) {
        result += energyParticle._energy;
    }
    return result;
}

bool IntegrationTestFramework::compare(Description left, Description right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(ObjectDescription left, ObjectDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(EnergyDescription left, EnergyDescription right) const
{
    return DescriptionTestDataFactory::get().compare(left, right);
}
