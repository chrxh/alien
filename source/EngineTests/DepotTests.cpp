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

protected:
    // Helper to create a depot creature with a generator that triggers it with a positive signal
    Description createDepotWithPositiveGenerator(float usableEnergy, float storedUsableEnergy = 0.0f, float storageLimit = 0.0f)
    {
        auto data = Description().addCreature(CreatureDescription().id(1).cells({
            CellDescription()
                .id(1)
                .pos({100.0f, 100.0f})
                .cellType(DepotDescription().storedUsableEnergy(storedUsableEnergy).storageLimit(storageLimit))
                .usableEnergy(usableEnergy),
            CellDescription().id(2).pos({101.0f, 100.0f}).cellType(GeneratorDescription().autoTriggerInterval(3).pulseType(GeneratorPulseType_Positive)),
        }));
        data.addConnection(1, 2);
        return data;
    }

    // Helper to create a depot creature with a generator that triggers it with a negative signal
    Description createDepotWithNegativeGenerator(float usableEnergy, float storedUsableEnergy = 0.0f)
    {
        // Using alternation with interval 0 produces -1.0f on first pulse since numPulses (0) is not < alternationInterval (0)
        auto data = Description().addCreature(CreatureDescription().id(1).cells({
            CellDescription().id(1).pos({100.0f, 100.0f}).cellType(DepotDescription().storedUsableEnergy(storedUsableEnergy)).usableEnergy(usableEnergy),
            CellDescription()
                .id(2)
                .pos({101.0f, 100.0f})
                .cellType(GeneratorDescription().autoTriggerInterval(3).pulseType(GeneratorPulseType_Alternation).alternationInterval(0)),
        }));
        data.addConnection(1, 2);
        return data;
    }
};

TEST_F(DepotTests, noSignal_noChange)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 20.0f;

    // Create depot without a generator - no signal will be sent
    auto data = Description().cells({
        CellDescription().id(1).pos({100.0f, 100.0f}).cellType(DepotDescription().storedUsableEnergy(50.0f)).usableEnergy(initialUsableEnergy),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);

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

    auto data = createDepotWithPositiveGenerator(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

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

    auto data = createDepotWithNegativeGenerator(normalCellEnergy, initialStoredEnergy);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

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

    auto data = createDepotWithPositiveGenerator(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been stored since usableEnergy <= normalCellEnergy
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, negativeSignal_noStoredEnergy_noRelease)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = createDepotWithNegativeGenerator(normalCellEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

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

    auto data = createDepotWithPositiveGenerator(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto origGenerator = data.getCellRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);
    auto actualGenerator = actualData.getCellRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(initialUsableEnergy + origGenerator._usableEnergy - SimulationParameters::depotEnergyTransferUnit, actualDepot._usableEnergy + actualGenerator._usableEnergy));
}

TEST_F(DepotTests, positiveSignal_reachedStorageLimit1)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 100.0f;
    auto storageLimit = 250.0f;
    auto initialStoredUsableEnergy = storageLimit - SimulationParameters::depotEnergyTransferUnit / 2;

    auto data = createDepotWithPositiveGenerator(initialUsableEnergy, initialStoredUsableEnergy, storageLimit);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto origGenerator = data.getCellRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);
    auto actualGenerator = actualData.getCellRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    // Energy should be capped at storageLimit
    EXPECT_TRUE(approxCompare(storageLimit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_reachedStorageLimit2)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 100.0f;
    auto storageLimit = SimulationParameters::depotStorageLimit * 2;  // Cell allows more than parameter limit
    auto initialStoredUsableEnergy = SimulationParameters::depotStorageLimit - SimulationParameters::depotEnergyTransferUnit / 2;

    auto data = createDepotWithPositiveGenerator(initialUsableEnergy, initialStoredUsableEnergy, storageLimit);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto origGenerator = data.getCellRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);
    auto actualGenerator = actualData.getCellRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    // Energy should be capped at SimulationParameters::depotstorageLimit
    EXPECT_TRUE(approxCompare(SimulationParameters::depotStorageLimit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, negativeSignal_energyTransferCapped)
{
    auto origDepotEnergy = _parameters.normalCellEnergy.value[0];
    // Provide more stored energy than depotEnergyTransferUnit
    auto initialStoredEnergy = 100.0f;

    auto data = createDepotWithNegativeGenerator(origDepotEnergy, initialStoredEnergy);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(4);  // Wait for generator to trigger

    auto origGenerator = data.getCellRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getCellRef(1);
    auto actualGenerator = actualData.getCellRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(
        initialStoredEnergy - SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(
        origDepotEnergy + origGenerator._usableEnergy + SimulationParameters::depotEnergyTransferUnit,
        actualDepot._usableEnergy + actualGenerator._usableEnergy));
}
