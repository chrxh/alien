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
};

TEST_F(MutationTests, dataMutation_genomeWithCellWithoutFunction_startPos)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes({CellGenomeDescription()});
    int genomePos = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(genomePos)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simController->testOnly_mutateData(1);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(genomePos, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, dataMutation_genomeWithCellWithoutFunction_endPos)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes({CellGenomeDescription()});
    int genomePos = toInt(genome.size());

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(genomePos)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simController->testOnly_mutateData(1);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(genomePos, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, dataMutation_genomeWithCellWithoutFunction_invalidPos)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes({CellGenomeDescription()});
    int genomePos = toInt(genome.size()) / 2;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(genomePos)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 1000; ++i) {
        _simController->testOnly_mutateData(1);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(genomePos, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, dataMutation_genomeWithMultpleCellsWithDifferentFunctions_startPos)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int genomePos = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(genomePos)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutateData(1);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(genomePos, actualConstructor.currentGenomePos);
}
