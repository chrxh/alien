#include "IntegrationTestFramework.h"

#include <boost/range/combine.hpp>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParameters.h>

#include <EngineTestData/DescTestDataFactory.h>

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
        _simulationFacade->setPreviewData(Desc());
        _simulationFacade->setCurrentTimestepForPreview(0);
        _simulationFacade->setCurrentTimestep(0);
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }
}

IntegrationTestFramework::~IntegrationTestFramework() {}

double IntegrationTestFramework::getEnergy(Desc const& data) const
{
    auto getDepotEnergy = [](ObjectDesc const& object) -> double {
        if (object.getObjectType() == ObjectType_Cell) {
            if (object.getCellRef().getCellType() == CellType_Depot) {
                auto const& depot = std::get<DepotDesc>(object.getCellRef()._cellType);
                return depot._storedUsableEnergy;
            }
        }
        return 0;
    };

    double result = 0;
    for (auto const& object : data._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            auto const& cell = object.getCellRef();
            auto reservedEnergy = cell._constructor.has_value() ? cell._constructor->_reservedEnergy : 0.0f;
            result += cell._usableEnergy + cell._rawEnergy + reservedEnergy + getDepotEnergy(object);
        } else if (object.getObjectType() == ObjectType_FreeCell) {
            result += object.getFreeCellRef()._energy;
        } else if (object.getObjectType() == ObjectType_Solid) {
            result += object.getSolidRef()._energy;
        } else if (object.getObjectType() == ObjectType_Fluid) {
            result += object.getFluidRef()._energy;
        }
    }
    for (auto const& energyParticle : data._energies) {
        result += energyParticle._energy;
    }
    return result;
}

bool IntegrationTestFramework::compare(Desc left, Desc right) const
{
    return DescTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(ObjectDesc left, ObjectDesc right) const
{
    return DescTestDataFactory::get().compare(left, right);
}

bool IntegrationTestFramework::compare(EnergyDesc left, EnergyDesc right) const
{
    return DescTestDataFactory::get().compare(left, right);
}
