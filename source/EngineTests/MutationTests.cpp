#include <cmath>

#include <algorithm>
#include <ranges>
#include <cstdlib>
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
    std::vector<int> const GenomeCellColors = {1, 4, 5};
    std::vector<uint8_t> createGenomeWithMultipleCellsWithDifferentFunctions() const
    {
        std::vector<uint8_t> subGenome;
        for (int i = 0; i < 15; ++i) {
            subGenome = GenomeDescriptionConverter::convertDescriptionToBytes(GenomeDescription{
                CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(GenomeCellColors[0]),
                CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(GenomeCellColors[1]),
                CellGenomeDescription().setColor(GenomeCellColors[2]),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeGenomeCopy()).setColor(GenomeCellColors[2]),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100)).setColor(GenomeCellColors[0]),
            });
        };
        return GenomeDescriptionConverter::convertDescriptionToBytes({
            CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(GenomeCellColors[0]),
            CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(GenomeCellColors[1]),
            CellGenomeDescription().setColor(GenomeCellColors[0]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeGenomeCopy()).setColor(GenomeCellColors[1]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)).setColor(GenomeCellColors[0]),
            CellGenomeDescription().setCellFunction(SensorGenomeDescription()).setColor(GenomeCellColors[2]),
            CellGenomeDescription().setCellFunction(NerveGenomeDescription()).setColor(GenomeCellColors[1]),
            CellGenomeDescription().setCellFunction(AttackerGenomeDescription()).setColor(GenomeCellColors[0]),
            CellGenomeDescription().setCellFunction(InjectorGenomeDescription().setGenome(subGenome)).setColor(GenomeCellColors[0]),
            CellGenomeDescription().setCellFunction(MuscleGenomeDescription()).setColor(GenomeCellColors[2]),
            CellGenomeDescription().setCellFunction(DefenderGenomeDescription()).setColor(GenomeCellColors[2]),
            CellGenomeDescription().setCellFunction(PlaceHolderGenomeDescription()).setColor(GenomeCellColors[0]),
        });
    }

    void rollout(GenomeDescription const& input, std::set<CellGenomeDescription>& result)
    {
        for (auto const& cell : input) {
            if (auto subGenome = cell.getSubGenome()) {
                auto subGenomeCells = GenomeDescriptionConverter::convertBytesToDescription(*subGenome);
                rollout(subGenomeCells, result);
                auto cellClone = cell;
                cellClone.setSubGenome({});
                result.insert(cellClone);
            } else {
                result.insert(cell);
            }
        }
    }

    bool compareDataMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (expectedCell.getCellFunctionType() != actualCell.getCellFunctionType()) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(*expectedCell.cellFunction);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(*actualCell.cellFunction);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareDataMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
                        return false;
                    }
                }
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(*expectedCell.cellFunction);
                auto actualInjector = std::get<InjectorGenomeDescription>(*actualCell.cellFunction);
                if (expectedInjector.isMakeGenomeCopy() != actualInjector.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedInjector.isMakeGenomeCopy()) {
                    if (!compareDataMutation(expectedInjector.getGenomeData(), actualInjector.getGenomeData())) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool compareNeuronDataMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (expectedCell.getCellFunctionType() != actualCell.getCellFunctionType()) {
                return false;
            }
            if (expectedCell.getCellFunctionType() != CellFunction_Neuron && expectedCell.getCellFunctionType() != CellFunction_Constructor
                && expectedCell.getCellFunctionType() != CellFunction_Injector && expectedCell != actualCell) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(*expectedCell.cellFunction);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(*actualCell.cellFunction);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareNeuronDataMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
                        return false;
                    }
                }
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(*expectedCell.cellFunction);
                auto actualInjector = std::get<InjectorGenomeDescription>(*actualCell.cellFunction);
                if (expectedInjector.isMakeGenomeCopy() != actualInjector.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedInjector.isMakeGenomeCopy()) {
                    if (!compareNeuronDataMutation(expectedInjector.getGenomeData(), actualInjector.getGenomeData())) {
                        return false;
                    }
                }
            }

        }
        return true;
    }

    bool compareCellFunctionMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        auto expectedGenome = GenomeDescriptionConverter::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverter::convertBytesToDescription(actual);
        if (expectedGenome.size() != actualGenome.size()) {
            return false;
        }
        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome, actualGenome)) {
            if (std::abs(expectedCell.referenceAngle - actualCell.referenceAngle) > NEAR_ZERO) {
                return false;
            }
            if (std::abs(expectedCell.energy- actualCell.energy) > NEAR_ZERO) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.numRequiredAdditionalConnections != actualCell.numRequiredAdditionalConnections) {
                return false;
            }
            if (expectedCell.executionOrderNumber != actualCell.executionOrderNumber) {
                return false;
            }
            if (expectedCell.inputExecutionOrderNumber != actualCell.inputExecutionOrderNumber) {
                return false;
            }
            if (expectedCell.outputBlocked != actualCell.outputBlocked) {
                return false;
            }
        }
        return true;
    }

    bool compareInsertMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after)
    {
        auto beforeGenome = GenomeDescriptionConverter::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverter::convertBytesToDescription(after);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(GenomeCellColors, cell.color) == GenomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& beforeCell : beforeGenome) {
            auto matchingAfterCells = afterGenome | std::views::filter([&beforeCell](auto const& afterCell) {
                auto beforeCellClone = beforeCell;
                auto afterCellClone = afterCell;
                beforeCellClone.cellFunction.reset();
                afterCellClone.cellFunction.reset();
                return beforeCellClone == afterCellClone;
            });
            if (matchingAfterCells.empty()) {
                return false;
            }
            if (beforeCell.getCellFunctionType() == CellFunction_Constructor || beforeCell.getCellFunctionType() == CellFunction_Injector) {
                auto matches = false;
                auto beforeSubGenome = beforeCell.getSubGenome();
                auto beforeIsMakeCopyGenome = beforeCell.isMakeGenomeCopy();
                for (auto const& afterCell : matchingAfterCells) {
                    auto afterIsMakeCopyGenome = afterCell.isMakeGenomeCopy();
                    if (beforeIsMakeCopyGenome && *beforeIsMakeCopyGenome && afterIsMakeCopyGenome && *afterIsMakeCopyGenome) {
                        matches = true;
                        break;
                    }
                    auto afterSubGenome = afterCell.getSubGenome();
                    if (beforeSubGenome && afterSubGenome) {
                        matches |= compareInsertMutation(*beforeSubGenome, *afterSubGenome);
                    }
                }
                if (!matches) {
                    return false;
                }
            }
        }
        return true;
    }

    bool compareDeleteMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after)
    {
        auto beforeGenome = GenomeDescriptionConverter::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverter::convertBytesToDescription(after);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(GenomeCellColors, cell.color) == GenomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& afterCell : afterGenome) {
            auto matchingBeforeCells = beforeGenome | std::views::filter([&afterCell](auto const& beforeCell) {
                                          auto beforeCellClone = beforeCell;
                                          auto afterCellClone = afterCell;
                                          beforeCellClone.cellFunction.reset();
                                          afterCellClone.cellFunction.reset();
                                          return beforeCellClone == afterCellClone;
                                      });
            if (matchingBeforeCells.empty()) {
                return false;
            }
            if (afterCell.getCellFunctionType() == CellFunction_Constructor || afterCell.getCellFunctionType() == CellFunction_Injector) {
                auto matches = false;
                auto afterSubGenome = afterCell.getSubGenome();
                auto afterIsMakeCopyGenome = afterCell.isMakeGenomeCopy();
                for (auto const& beforeCell : matchingBeforeCells) {
                    auto beforeIsMakeCopyGenome = beforeCell.isMakeGenomeCopy();
                    if (afterIsMakeCopyGenome && *afterIsMakeCopyGenome && beforeIsMakeCopyGenome && *beforeIsMakeCopyGenome) {
                        matches = true;
                        break;
                    }
                    auto beforeSubGenome = beforeCell.getSubGenome();
                    if (beforeSubGenome && beforeSubGenome) {
                        matches |= compareDeleteMutation(*beforeSubGenome, *beforeSubGenome);
                    }
                }
                if (!matches) {
                    return false;
                }
            }
        }
        return true;
    }

    bool compareTranslateMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after)
    {
        auto beforeGenome = GenomeDescriptionConverter::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverter::convertBytesToDescription(after);

        std::set<CellGenomeDescription> beforeGenomeRollout;
        rollout(beforeGenome, beforeGenomeRollout);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);

        return beforeGenomeRollout == afterGenomeRollout;
    }
};

TEST_F(MutationTests, dataMutation_startPos)
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

TEST_F(MutationTests, dataMutation_endPos)
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

TEST_F(MutationTests, dataMutation_invalidPos)
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

TEST_F(MutationTests, neuronDataMutation)
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

TEST_F(MutationTests, cellFunctionMutation)
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

TEST_F(MutationTests, insertMutation_emptyGenome)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto cellColor = 3;
    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription()).setExecutionOrderNumber(0).setColor(cellColor)});

    _simController->setSimulationData(data);
    _simController->testOnly_mutate(1, MutationType::Insertion);

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);

    auto actualGenomeDescription = GenomeDescriptionConverter::convertBytesToDescription(actualConstructor.genome);
    EXPECT_EQ(1, actualGenomeDescription.size());
    EXPECT_EQ(cellColor, actualGenomeDescription.front().color);
}

TEST_F(MutationTests, insertMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Insertion);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, deleteMutation_eraseSmallGenome)
{
    auto genome = GenomeDescriptionConverter::convertDescriptionToBytes({
        CellGenomeDescription().setCellFunction(NeuronGenomeDescription()),
    });

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    _simController->testOnly_mutate(1, MutationType::Deletion);

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(actualConstructor.genome.empty());
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_preserveSelfReplication)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    auto afterGenome = GenomeDescriptionConverter::convertBytesToDescription(actualConstructor.genome);

    std::set<CellGenomeDescription> afterGenomeRollout;
    rollout(afterGenome, afterGenomeRollout);
    for (auto const& cell : afterGenomeRollout) {
        auto cellFunctionType = cell.getCellFunctionType();
        EXPECT_TRUE(cellFunctionType == CellFunction_Constructor || cellFunctionType == CellFunction_Injector);
    }
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_changeSelfReplication)
{
    auto parameters = _parameters;
    parameters.cellFunctionConstructorMutationSelfReplication = true;
    _simController->setSimulationParameters(parameters);

    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_EQ(0, actualConstructor.genome.size());
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, deleteMutation_partiallyEraseGenome)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simController->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDeleteMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, duplicateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simController->testOnly_mutate(1, MutationType::Duplication);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.currentGenomePos);
}

TEST_F(MutationTests, translateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setCurrentGenomePos(0)).setExecutionOrderNumber(0)});

    _simController->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simController->testOnly_mutate(1, MutationType::Translation);
    }

    auto actualData = _simController->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareTranslateMutation(genome, actualConstructor.genome));
}
