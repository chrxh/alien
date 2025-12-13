#include <gtest/gtest.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DepotTests : public IntegrationTestFramework
{
public:
    DepotTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~DepotTests() = default;
};

TEST_F(DepotTests, noSignal_noChange)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(DepotDescription().storedUsableEnergy(50.0f)).usableEnergy(normalCellEnergy + 20.0f),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origDepot = data.getCellRef(1);
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(origDepot._usableEnergy, actualDepot._usableEnergy));
    EXPECT_TRUE(approxCompare(50.0f, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_storeEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 20.0f;

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(0.0f))
            .usableEnergy(initialUsableEnergy)
            .signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy should have been transferred to storage
    EXPECT_TRUE(actualDepot._usableEnergy < initialUsableEnergy - NEAR_ZERO);
    EXPECT_TRUE(std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy > NEAR_ZERO);
}

TEST_F(DepotTests, negativeSignal_releaseEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialStoredEnergy = 50.0f;

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(initialStoredEnergy))
            .usableEnergy(normalCellEnergy)
            .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy should have been released from storage
    EXPECT_TRUE(actualDepot._usableEnergy > normalCellEnergy + NEAR_ZERO);
    EXPECT_TRUE(std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy < initialStoredEnergy - NEAR_ZERO);
}

TEST_F(DepotTests, positiveSignal_usableEnergyBelowNormal_noStore)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy - 10.0f;

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(0.0f))
            .usableEnergy(initialUsableEnergy)
            .signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been stored since usableEnergy <= normalCellEnergy
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, negativeSignal_noStoredEnergy_noRelease)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(0.0f))
            .usableEnergy(normalCellEnergy)
            .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been released since storedUsableEnergy was 0
    EXPECT_TRUE(approxCompare(normalCellEnergy, actualDepot._usableEnergy));
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_energyTransferCapped)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    // Provide much more energy than depotEnergyTransferUnit
    auto initialUsableEnergy = normalCellEnergy + 100.0f;

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(0.0f))
            .usableEnergy(initialUsableEnergy)
            .signalAndState({1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(initialUsableEnergy - SimulationParameters::depotEnergyTransferUnit, actualDepot._usableEnergy));
}

TEST_F(DepotTests, negativeSignal_energyTransferCapped)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    // Provide more stored energy than depotEnergyTransferUnit
    auto initialStoredEnergy = 100.0f;

    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(initialStoredEnergy))
            .usableEnergy(normalCellEnergy)
            .signalAndState({-1, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(
        initialStoredEnergy - SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(normalCellEnergy + SimulationParameters::depotEnergyTransferUnit, actualDepot._usableEnergy));
}

TEST_F(DepotTests, signalBelowThreshold_noChange)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 20.0f;

    // Signal magnitude is 0.05, which is below TRIGGER_THRESHOLD
    auto data = Description().cells({
        CellDescription()
            .id(1)
            .pos({100.0f, 100.0f})
            .cellType(DepotDescription().storedUsableEnergy(0.0f))
            .usableEnergy(initialUsableEnergy)
            .signalAndState({0.05f, 0, 0, 0, 0, 0, 0, 0}),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been stored since signal is below threshold
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}
