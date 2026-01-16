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
    Description createDepotWithIncomingPositiveSignal(float usableEnergy, float storedUsableEnergy = 0.0f, float storageLimit = 200.0f)
    {
        auto data = Description().addCreature({
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(DepotDescription().storedUsableEnergy(storedUsableEnergy).storageLimit(storageLimit)).usableEnergy(usableEnergy)),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({1, 0, 0, 0, 0, 0, 0, 0})),
        }, CreatureDescription().id(1));
        data.addConnection(1, 2);
        return data;
    }

    Description createDepotWithIncomingNegativeSignal(float usableEnergy, float storedUsableEnergy = 0.0f)
    {
        // Using alternation with interval 0 produces -1.0f on first pulse since numPulses (0) is not < alternationInterval (0)
        auto data = Description().addCreature({
            ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(DepotDescription().storedUsableEnergy(storedUsableEnergy)).usableEnergy(usableEnergy)),
            ObjectDescription().id(2).pos({101.0f, 100.0f}).type(CellDescription().signalAndState({-1, 0, 0, 0, 0, 0, 0, 0})),
        }, CreatureDescription().id(1));
        data.addConnection(1, 2);
        return data;
    }
};

TEST_F(DepotTests, noSignal_noChange)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 20.0f;

    // Create depot without a cell carrying a signel => no signal will be sent
    auto data = Description().objects({
        ObjectDescription().id(1).pos({100.0f, 100.0f}).type(CellDescription().cellType(DepotDescription().storedUsableEnergy(50.0f)).usableEnergy(initialUsableEnergy)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origDepot = data.getObjectRef(1);
    auto actualDepot = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(origDepot.getCellRef()._usableEnergy, actualDepot.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(50.0f, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_storeEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 20.0f;

    auto data = createDepotWithIncomingPositiveSignal(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy should have been transferred to storage
    EXPECT_TRUE(actualDepot.getCellRef()._usableEnergy < initialUsableEnergy - NEAR_ZERO);
    EXPECT_TRUE(std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy > NEAR_ZERO);
}

TEST_F(DepotTests, negativeSignal_releaseEnergy)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialStoredEnergy = 50.0f;

    auto data = createDepotWithIncomingNegativeSignal(normalCellEnergy, initialStoredEnergy);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy should have been released from storage
    EXPECT_TRUE(actualDepot.getCellRef()._usableEnergy > normalCellEnergy + NEAR_ZERO);
    EXPECT_TRUE(std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy < initialStoredEnergy - NEAR_ZERO);
}

TEST_F(DepotTests, positiveSignal_usableEnergyBelowNormal_noStore)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy - 10.0f;

    auto data = createDepotWithIncomingPositiveSignal(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been stored since usableEnergy <= normalCellEnergy
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, negativeSignal_noStoredEnergy_noRelease)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];

    auto data = createDepotWithIncomingNegativeSignal(normalCellEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // No energy should have been released since storedUsableEnergy was 0
    EXPECT_TRUE(approxCompare(normalCellEnergy, actualDepot.getCellRef()._usableEnergy));
    EXPECT_TRUE(approxCompare(0.0f, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_energyTransferCapped)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    // Provide much more energy than depotEnergyTransferUnit
    auto initialUsableEnergy = normalCellEnergy + 100.0f;

    auto data = createDepotWithIncomingPositiveSignal(initialUsableEnergy, 0.0f);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto origOtherCell = data.getObjectRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(
        initialUsableEnergy + origOtherCell.getCellRef()._usableEnergy - SimulationParameters::depotEnergyTransferUnit,
        actualDepot.getCellRef()._usableEnergy + actualOtherCell.getCellRef()._usableEnergy));
}

TEST_F(DepotTests, positiveSignal_reachedStorageLimit1)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 100.0f;
    auto storageLimit = 250.0f;
    auto initialStoredUsableEnergy = storageLimit - SimulationParameters::depotEnergyTransferUnit / 2;

    auto data = createDepotWithIncomingPositiveSignal(initialUsableEnergy, initialStoredUsableEnergy, storageLimit);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto origOtherCell = data.getObjectRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    // Energy should be capped at storageLimit
    EXPECT_TRUE(approxCompare(storageLimit, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, positiveSignal_reachedStorageLimit2)
{
    auto normalCellEnergy = _parameters.normalCellEnergy.value[0];
    auto initialUsableEnergy = normalCellEnergy + 100.0f;
    auto storageLimit = SimulationParameters::depotStorageLimit * 2;  // Cell allows more than parameter limit
    auto initialStoredUsableEnergy = SimulationParameters::depotStorageLimit - SimulationParameters::depotEnergyTransferUnit / 2;

    auto data = createDepotWithIncomingPositiveSignal(initialUsableEnergy, initialStoredUsableEnergy, storageLimit);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto origOtherCell = data.getObjectRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));

    // Energy should be capped at SimulationParameters::depotstorageLimit
    EXPECT_TRUE(approxCompare(SimulationParameters::depotStorageLimit, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
}

TEST_F(DepotTests, negativeSignal_energyTransferCapped)
{
    auto origDepotEnergy = _parameters.normalCellEnergy.value[0];
    // Provide more stored energy than depotEnergyTransferUnit
    auto initialStoredEnergy = 100.0f;

    auto data = createDepotWithIncomingNegativeSignal(origDepotEnergy, initialStoredEnergy);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto origOtherCell = data.getObjectRef(2);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualDepot = actualData.getObjectRef(1);
    auto actualOtherCell = actualData.getObjectRef(2);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    // Energy transfer should be capped at depotEnergyTransferUnit
    EXPECT_TRUE(approxCompare(
        initialStoredEnergy - SimulationParameters::depotEnergyTransferUnit, std::get<DepotDescription>(actualDepot.getCellRef()._cellType)._storedUsableEnergy));
    EXPECT_TRUE(approxCompare(
        origDepotEnergy + origOtherCell.getCellRef()._usableEnergy + SimulationParameters::depotEnergyTransferUnit,
        actualDepot.getCellRef()._usableEnergy + actualOtherCell.getCellRef()._usableEnergy));
}
