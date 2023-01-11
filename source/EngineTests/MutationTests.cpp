#include <cmath>

#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

#include "IntegrationTestFramework.h"

class MutationTests : public IntegrationTestFramework
{
public:
    MutationTests()
        : IntegrationTestFramework()
    {}

    ~MutationTests() = default;

protected:
    std::vector<uint8_t> createGenomeWithMultipleCellsWithDifferentFunctions() const
    {
        std::vector<uint8_t> dummyData(8, 0);
        return GenomeDescriptionConverter::convertDescriptionToBytes({
            CellGenomeDescription().setCellFunction(NeuronGenomeDescription()),
            CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()),
            CellGenomeDescription(),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeGenomeCopy()),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(dummyData)),
            CellGenomeDescription().setCellFunction(SensorGenomeDescription()),
            CellGenomeDescription().setCellFunction(NerveGenomeDescription()),
            CellGenomeDescription().setCellFunction(AttackerGenomeDescription()),
            CellGenomeDescription().setCellFunction(InjectorGenomeDescription()),
            CellGenomeDescription().setCellFunction(MuscleGenomeDescription()),
            CellGenomeDescription().setCellFunction(PlaceHolderGenomeDescription1()),
            CellGenomeDescription().setCellFunction(PlaceHolderGenomeDescription2()),
        });
    }

    bool compareDataMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected, _parameters);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual, _parameters);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (expectedCell.getCellFunctionType() != actualCell.getCellFunctionType()) {
                return false;
            }
        }
        return true;
    }

    bool compareNeuronDataMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected, _parameters);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual, _parameters);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (expectedCell.getCellFunctionType() != actualCell.getCellFunctionType()) {
                return false;
            }
            if (expectedCell.getCellFunctionType() != CellFunction_Neuron && expectedCell != actualCell) {
                return false;
            }
        }
        return true;
    }

    bool compareCellFunctionMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected, _parameters);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual, _parameters);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }
        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (std::abs(expectedCell.referenceDistance - actualCell.referenceDistance) > NEAR_ZERO) {
                return false;
            }
            if (std::abs(expectedCell.referenceAngle - actualCell.referenceAngle) > NEAR_ZERO) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.maxConnections != actualCell.maxConnections) {
                return false;
            }
            if (expectedCell.executionOrderNumber != actualCell.executionOrderNumber) {
                return false;
            }
            if (expectedCell.inputBlocked != actualCell.inputBlocked) {
                return false;
            }
            if (expectedCell.outputBlocked != actualCell.outputBlocked) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(MutationTests, mutateData_startPos)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(byteIndex)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Data);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, mutateData_endPos)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = toInt(genome.size());

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(byteIndex)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Data);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, mutateData_invalidPos)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = toInt(genome.size()) / 2;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(byteIndex)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Data);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, mutateNeuronData)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(byteIndex)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::NeuronData);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareNeuronDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, mutateCellFunction)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    auto cellIndex = 7;
    int byteIndex = GenomeDescriptionConverter::convertCellIndexToByteIndex(genome, cellIndex);

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(byteIndex)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::CellFunction);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareCellFunctionMutation(genome, actualConstructor.genome));
    auto actualCellIndex = GenomeDescriptionConverter::convertByteIndexToCellIndex(actualConstructor.genome, actualConstructor.currentGenomePos);
    EXPECT_EQ(cellIndex, actualCellIndex);
}