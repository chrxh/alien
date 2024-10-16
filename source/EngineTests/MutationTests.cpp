#include <cmath>

#include <algorithm>
#include <ranges>
#include <cstdlib>
#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionService.h"

#include "IntegrationTestFramework.h"

class MutationTests : public IntegrationTestFramework
{
public:
    MutationTests()
        : IntegrationTestFramework()
    {}

    ~MutationTests() = default;

protected:
    std::vector<int> const genomeCellColors = {1, 4, 5};
    std::vector<uint8_t> createGenomeWithMultipleCellsWithDifferentFunctions() const
    {
        std::vector<uint8_t> subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 14; ++i) {
            subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(genomeCellColors[0]),
                CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(genomeCellColors[1]),
                CellGenomeDescription().setColor(genomeCellColors[2]),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[2]),
                CellGenomeDescription()
                    .setCellFunction(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100))
                    .setColor(genomeCellColors[0]),
            }));
        }
        return GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(SensorGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellFunction(NerveGenomeDescription()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setCellFunction(AttackerGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(InjectorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(MuscleGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellFunction(DefenderGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellFunction(ReconnectorGenomeDescription()).setColor(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColorPerSubgenome() const
    {
        std::vector<uint8_t> subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            auto color = genomeCellColors[i % genomeCellColors.size()];
            subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(color),
                CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(color),
                CellGenomeDescription().setColor(color),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
                CellGenomeDescription()
                    .setCellFunction(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100))
                    .setColor(color),
            }));
        };
        return GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(SensorGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(NerveGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(AttackerGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(InjectorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(MuscleGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(DefenderGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellFunction(ReconnectorGenomeDescription()).setColor(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColor() const
    {
        auto color = genomeCellColors[0];
        std::vector<uint8_t> subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            subGenome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(color),
                CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(color),
                CellGenomeDescription().setColor(color),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
                CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100)).setColor(color),
            }));
        };
        return GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellFunction(NeuronGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(TransmitterGenomeDescription()).setColor(color),
            CellGenomeDescription().setColor(color),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
            CellGenomeDescription().setCellFunction(ConstructorGenomeDescription().setGenome(subGenome)).setColor(color),
            CellGenomeDescription().setCellFunction(SensorGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(NerveGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(AttackerGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(InjectorGenomeDescription().setGenome(subGenome)).setColor(color),
            CellGenomeDescription().setCellFunction(MuscleGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(DefenderGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellFunction(ReconnectorGenomeDescription()).setColor(color),
        }));
    }

    void rollout(GenomeDescription const& input, std::set<CellGenomeDescription>& result)
    {
        for (auto const& cell : input.cells) {
            if (auto subGenomeBytes = cell.getGenome()) {
                auto subGenome = GenomeDescriptionService::convertBytesToDescription(*subGenomeBytes);
                rollout(subGenome, result);
                auto cellClone = cell;
                cellClone.setGenome({});
                result.insert(cellClone);
            } else {
                result.insert(cell);
            }
        }
    }

    bool comparePropertiesMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionService::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::convertBytesToDescription(actual);
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
            if (expectedCell.getCellFunctionType() != actualCell.getCellFunctionType()) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.referenceAngle != actualCell.referenceAngle) {
                return false;
            }
            if (expectedCell.numRequiredAdditionalConnections != actualCell.numRequiredAdditionalConnections) {
                return false;
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(*expectedCell.cellFunction);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(*actualCell.cellFunction);
                if (expectedConstructor.constructionAngle1 != actualConstructor.constructionAngle1) {
                    return false;
                }
                if (expectedConstructor.constructionAngle2 != actualConstructor.constructionAngle2) {
                    return false;
                }
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!comparePropertiesMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
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
                    if (!comparePropertiesMutation(expectedInjector.getGenomeData(), actualInjector.getGenomeData())) {
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
        auto expectedGenome = GenomeDescriptionService::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::convertBytesToDescription(actual);
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
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

    bool compareGeometryMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionService::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::convertBytesToDescription(actual);
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        auto createCompareClone = [](CellGenomeDescription const& cell) {
            auto clone = cell;
            clone.referenceAngle = 0;
            clone.numRequiredAdditionalConnections = 0;
            if (clone.getCellFunctionType() == CellFunction_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(*clone.cellFunction);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor.genome = {};
                }
            }
            if (clone.getCellFunctionType() == CellFunction_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(*clone.cellFunction);
                if (!injector.isMakeGenomeCopy()) {
                    injector.genome = {};
                }
            }
            return clone;
        };

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
            if (createCompareClone(expectedCell) != createCompareClone(actualCell)) {
                return false;
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(*expectedCell.cellFunction);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(*actualCell.cellFunction);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareGeometryMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
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
                    if (!compareGeometryMutation(expectedInjector.getGenomeData(), actualInjector.getGenomeData())) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool compareIndividualGeometryMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        if (expected.size() != actual.size()) {
            return false;
        }
        auto expectedGenome = GenomeDescriptionService::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::convertBytesToDescription(actual);
        expectedGenome.header.shape = ConstructionShape_Custom; //compare all expect shape
        actualGenome.header.shape = ConstructionShape_Custom;
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        auto createCompareClone = [](CellGenomeDescription const& cell) {
            auto clone = cell;
            clone.referenceAngle = 0;
            clone.numRequiredAdditionalConnections = 0;
            if (clone.getCellFunctionType() == CellFunction_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(*clone.cellFunction);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor.genome = {};
                }
                constructor.constructionAngle1 = 0;
                constructor.constructionAngle2 = 0;
            }
            if (clone.getCellFunctionType() == CellFunction_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(*clone.cellFunction);
                if (!injector.isMakeGenomeCopy()) {
                    injector.genome = {};
                }
            }
            return clone;
        };

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
            if (createCompareClone(expectedCell) != createCompareClone(actualCell)) {
                return false;
            }
            if (expectedCell.getCellFunctionType() == CellFunction_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(*expectedCell.cellFunction);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(*actualCell.cellFunction);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareIndividualGeometryMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
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
                    if (!compareIndividualGeometryMutation(expectedInjector.getGenomeData(), actualInjector.getGenomeData())) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool compareCellFunctionMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        auto expectedGenome = GenomeDescriptionService::convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::convertBytesToDescription(actual);
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }
        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
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
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(genomeCellColors, cell.color) == genomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& beforeCell : beforeGenome.cells) {
            auto matchingAfterCells = afterGenome.cells | std::views::filter([&beforeCell](auto const& afterCell) {
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
                auto beforeSubGenome = beforeCell.getGenome();
                auto beforeIsMakeCopyGenome = beforeCell.isMakeGenomeCopy();
                for (auto const& afterCell : matchingAfterCells) {
                    auto afterIsMakeCopyGenome = afterCell.isMakeGenomeCopy();
                    if (beforeIsMakeCopyGenome && *beforeIsMakeCopyGenome && afterIsMakeCopyGenome && *afterIsMakeCopyGenome) {
                        matches = true;
                        break;
                    }
                    auto afterSubGenome = afterCell.getGenome();
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
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(genomeCellColors, cell.color) == genomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& afterCell : afterGenome.cells) {
            auto matchingBeforeCells = beforeGenome.cells | std::views::filter([&afterCell](auto const& beforeCell) {
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
                auto afterSubGenome = afterCell.getGenome();
                auto afterIsMakeCopyGenome = afterCell.isMakeGenomeCopy();
                for (auto const& beforeCell : matchingBeforeCells) {
                    auto beforeIsMakeCopyGenome = beforeCell.isMakeGenomeCopy();
                    if (afterIsMakeCopyGenome && *afterIsMakeCopyGenome && beforeIsMakeCopyGenome && *beforeIsMakeCopyGenome) {
                        matches = true;
                        break;
                    }
                    auto beforeSubGenome = beforeCell.getGenome();
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
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);

        std::set<CellGenomeDescription> beforeGenomeRollout;
        rollout(beforeGenome, beforeGenomeRollout);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);

        return beforeGenomeRollout == afterGenomeRollout;
    }

    bool compareCellColorMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after, std::set<int> const& allowedColors)
    {
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            beforeCellClone.cellFunction = std::nullopt;
            afterCellClone.color = 0;
            afterCellClone.cellFunction = std::nullopt;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (!allowedColors.contains(afterCell.color)) {
                return false;
            }
            if (beforeCell.getCellFunctionType() == CellFunction_Constructor || beforeCell.getCellFunctionType() == CellFunction_Injector) {
                auto beforeSubGenome = beforeCell.getGenome();
                auto afterSubGenome = afterCell.getGenome();
                if (beforeSubGenome && afterSubGenome) {
                    if (!compareCellColorMutation(*beforeSubGenome, *afterSubGenome, allowedColors)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool compareSubgenomeColorMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after, std::set<int> const& allowedColors)
    {
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        std::optional<int> uniformColor;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            beforeCellClone.cellFunction = std::nullopt;
            afterCellClone.color = 0;
            afterCellClone.cellFunction = std::nullopt;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (!allowedColors.contains(afterCell.color)) {
                return false;
            }
            if (uniformColor && afterCell.color != *uniformColor) {
                return false;
            }
            uniformColor = afterCell.color;
            if (beforeCell.getCellFunctionType() == CellFunction_Constructor || beforeCell.getCellFunctionType() == CellFunction_Injector) {
                auto beforeSubGenome = beforeCell.getGenome();
                auto afterSubGenome = afterCell.getGenome();
                if (beforeSubGenome && afterSubGenome) {
                    if (!compareSubgenomeColorMutation(*beforeSubGenome, *afterSubGenome, allowedColors)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool compareGenomeColorMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after, std::optional<int> const& allowedColor)
    {
        auto beforeGenome = GenomeDescriptionService::convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        int uniformColor = allowedColor ? *allowedColor : afterGenome.cells.at(0).color;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            beforeCellClone.cellFunction = std::nullopt;
            afterCellClone.color = 0;
            afterCellClone.cellFunction = std::nullopt;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (afterCell.color != uniformColor) {
                return false;
            }
            uniformColor = afterCell.color;
            if (beforeCell.getCellFunctionType() == CellFunction_Constructor || beforeCell.getCellFunctionType() == CellFunction_Injector) {
                auto beforeSubGenome = beforeCell.getGenome();
                auto afterSubGenome = afterCell.getGenome();
                if (beforeSubGenome && afterSubGenome) {
                    if (!compareGenomeColorMutation(*beforeSubGenome, *afterSubGenome, uniformColor)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
};

TEST_F(MutationTests, propertiesMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Properties);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(comparePropertiesMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, neuronDataMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::NeuronData);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareNeuronDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, geometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Geometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareGeometryMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, individualGeometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CustomGeometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareIndividualGeometryMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, cellFunctionMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(3)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellFunction);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareCellFunctionMutation(genome, actualConstructor.genome));
    EXPECT_EQ(3, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, insertMutation_emptyGenome)
{
    auto cellColor = 3;
    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription()).setExecutionOrderNumber(0).setColor(cellColor)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Insertion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);

    auto actualGenomeDescription = GenomeDescriptionService::convertBytesToDescription(actualConstructor.genome);
    EXPECT_EQ(1, actualGenomeDescription.cells.size());
    EXPECT_EQ(cellColor, actualGenomeDescription.cells.front().color);
}

TEST_F(MutationTests, insertMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells({CellDescription()
                                                .setId(1)
                                                .setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))
                                                .setExecutionOrderNumber(0)
                                                .setColor(genomeCellColors[0])});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Insertion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseSmallGenome)
{
    auto genome = GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription().setCells({
        CellGenomeDescription().setCellFunction(NeuronGenomeDescription()),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Deletion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_EQ(GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor.genome.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_preserveSelfReplication)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    auto afterGenome = GenomeDescriptionService::convertBytesToDescription(actualConstructor.genome);

    std::set<CellGenomeDescription> afterGenomeRollout;
    rollout(afterGenome, afterGenomeRollout);
    for (auto const& cell : afterGenomeRollout) {
        auto cellFunctionType = cell.getCellFunctionType();
        EXPECT_TRUE(cellFunctionType == CellFunction_Constructor || cellFunctionType == CellFunction_Injector);
    }
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_changeSelfReplication)
{
    _parameters.cellFunctionConstructorMutationSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);

    EXPECT_EQ(GenomeDescriptionService::convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor.genome.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_partiallyEraseGenome)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareDeleteMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, duplicateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Duplication);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, translateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Translation);
    }
    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);
    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareTranslateMutation(genome, actualConstructor.genome));
}

TEST_F(MutationTests, cellColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellFunctionConstructorMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellFunctionConstructorMutationColorTransitions[0][3] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[0][5] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][2] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColorPerSubgenome();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareCellColorMutation(genome, actualConstructor.genome, {1, 2, 4, 5}));
}

TEST_F(MutationTests, subgenomeColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellFunctionConstructorMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellFunctionConstructorMutationColorTransitions[0][3] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[0][5] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][2] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColorPerSubgenome();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::SubgenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareSubgenomeColorMutation(genome, actualConstructor.genome, {1, 2, 4, 5}));
}

TEST_F(MutationTests, genomeColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellFunctionConstructorMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellFunctionConstructorMutationColorTransitions[0][3] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[0][5] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][2] = true;
    _parameters.cellFunctionConstructorMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColor();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellFunction(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0)).setExecutionOrderNumber(0)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::GenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(*actualCellById.at(1).cellFunction);
    EXPECT_TRUE(compareGenomeColorMutation(genome, actualConstructor.genome, std::nullopt));
}
