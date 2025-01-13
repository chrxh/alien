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
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.baseValues.cellCopyMutationNeuronData[i] = 1;
            _parameters.baseValues.cellCopyMutationCellProperties[i] = 1;
            _parameters.baseValues.cellCopyMutationCellType[i] = 1;
            _parameters.baseValues.cellCopyMutationGeometry[i] = 1;
            _parameters.baseValues.cellCopyMutationCustomGeometry[i] = 1;
            _parameters.baseValues.cellCopyMutationInsertion[i] = 1;
            _parameters.baseValues.cellCopyMutationDeletion[i] = 1;
            _parameters.baseValues.cellCopyMutationTranslation[i] = 1;
            _parameters.baseValues.cellCopyMutationDuplication[i] = 1;
            _parameters.baseValues.cellCopyMutationCellColor[i] = 1;
            _parameters.baseValues.cellCopyMutationSubgenomeColor[i] = 1;
            _parameters.baseValues.cellCopyMutationGenomeColor[i] = 1;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~MutationTests() = default;

protected:
    std::vector<int> const genomeCellColors = {1, 4, 5};
    std::vector<uint8_t> createGenomeWithMultipleCellsWithDifferentFunctions() const
    {
        std::vector<uint8_t> subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 14; ++i) {
            subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(genomeCellColors[0]),
                CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(genomeCellColors[1]),
                CellGenomeDescription().setColor(genomeCellColors[2]),
                CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[2]),
                CellGenomeDescription()
                    .setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100))
                    .setColor(genomeCellColors[0]),
            }));
        }
        return GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(SensorGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellTypeData(OscillatorGenomeDescription()).setColor(genomeCellColors[1]),
            CellGenomeDescription().setCellTypeData(AttackerGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(InjectorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(MuscleGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellTypeData(DefenderGenomeDescription()).setColor(genomeCellColors[2]),
            CellGenomeDescription().setCellTypeData(ReconnectorGenomeDescription()).setColor(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColorPerSubgenome() const
    {
        std::vector<uint8_t> subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            auto color = genomeCellColors[i % genomeCellColors.size()];
            subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(color),
                CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(color),
                CellGenomeDescription().setColor(color),
                CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
                CellGenomeDescription()
                    .setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100))
                    .setColor(color),
            }));
        };
        return GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(SensorGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(OscillatorGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(AttackerGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(InjectorGenomeDescription().setGenome(subGenome)).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(MuscleGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(DefenderGenomeDescription()).setColor(genomeCellColors[0]),
            CellGenomeDescription().setCellTypeData(ReconnectorGenomeDescription()).setColor(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColor() const
    {
        auto color = genomeCellColors[0];
        std::vector<uint8_t> subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            subGenome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
                CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(color),
                CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(color),
                CellGenomeDescription().setColor(color),
                CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
                CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome).setMode(std::rand() % 100)).setColor(color),
            }));
        };
        return GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
            CellGenomeDescription().setCellTypeData(BaseGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(DepotGenomeDescription()).setColor(color),
            CellGenomeDescription().setColor(color),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()).setColor(color),
            CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setGenome(subGenome)).setColor(color),
            CellGenomeDescription().setCellTypeData(SensorGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(OscillatorGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(AttackerGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(InjectorGenomeDescription().setGenome(subGenome)).setColor(color),
            CellGenomeDescription().setCellTypeData(MuscleGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(DefenderGenomeDescription()).setColor(color),
            CellGenomeDescription().setCellTypeData(ReconnectorGenomeDescription()).setColor(color),
        }));
    }

    void rollout(GenomeDescription const& input, std::set<CellGenomeDescription>& result)
    {
        for (auto const& cell : input.cells) {
            if (auto subGenomeBytes = cell.getGenome()) {
                auto subGenome = GenomeDescriptionService::get().convertBytesToDescription(*subGenomeBytes);
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
        auto expectedGenome = GenomeDescriptionService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actual);
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
            if (expectedCell.getCellType() != actualCell.getCellType()) {
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
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell.cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell.cellTypeData);
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
            if (expectedCell.getCellType() == CellType_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell.cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell.cellTypeData);
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
        auto expectedGenome = GenomeDescriptionService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actual);
        if (expectedGenome.header != actualGenome.header) {
            return false;
        }
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome.cells, actualGenome.cells)) {
            if (expectedCell.getCellType() != actualCell.getCellType()) {
                return false;
            }
            if (expectedCell.getCellType() != CellType_Base && expectedCell.getCellType() != CellType_Constructor
                && expectedCell.getCellType() != CellType_Injector && expectedCell != actualCell) {
                return false;
            }
            if (expectedCell.color != actualCell.color) {
                return false;
            }
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell.cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell.cellTypeData);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareNeuronDataMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
                        return false;
                    }
                }
            }
            if (expectedCell.getCellType() == CellType_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell.cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell.cellTypeData);
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
        auto expectedGenome = GenomeDescriptionService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actual);
        if (expectedGenome.cells.size() != actualGenome.cells.size()) {
            return false;
        }

        auto createCompareClone = [](CellGenomeDescription const& cell) {
            auto clone = cell;
            clone.referenceAngle = 0;
            clone.numRequiredAdditionalConnections = 0;
            if (clone.getCellType() == CellType_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(clone.cellTypeData);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor.genome = {};
                }
            }
            if (clone.getCellType() == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(clone.cellTypeData);
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
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell.cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell.cellTypeData);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareGeometryMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
                        return false;
                    }
                }
            }
            if (expectedCell.getCellType() == CellType_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell.cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell.cellTypeData);
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
        auto expectedGenome = GenomeDescriptionService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actual);
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
            if (clone.getCellType() == CellType_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(clone.cellTypeData);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor.genome = {};
                }
                constructor.constructionAngle1 = 0;
                constructor.constructionAngle2 = 0;
            }
            if (clone.getCellType() == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(clone.cellTypeData);
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
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell.cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell.cellTypeData);
                if (expectedConstructor.isMakeGenomeCopy() != actualConstructor.isMakeGenomeCopy()) {
                    return false;
                }
                if (!expectedConstructor.isMakeGenomeCopy()) {
                    if (!compareIndividualGeometryMutation(expectedConstructor.getGenomeData(), actualConstructor.getGenomeData())) {
                        return false;
                    }
                }
            }
            if (expectedCell.getCellType() == CellType_Injector) {
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell.cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell.cellTypeData);
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

    bool compareCellTypeMutation(std::vector<uint8_t> const& expected, std::vector<uint8_t> const& actual)
    {
        auto expectedGenome = GenomeDescriptionService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actual);
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
        }
        return true;
    }

    bool compareInsertMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after)
    {
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);
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
                                          afterCellClone.cellTypeData = beforeCellClone.cellTypeData;
                                          return beforeCellClone == afterCellClone;
                                      });
            if (matchingAfterCells.empty()) {
                return false;
            }
            if (beforeCell.getCellType() == CellType_Constructor || beforeCell.getCellType() == CellType_Injector) {
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
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);
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
                                           afterCellClone.cellTypeData = beforeCellClone.cellTypeData;
                                           return beforeCellClone == afterCellClone;
                                       });
            if (matchingBeforeCells.empty()) {
                return false;
            }
            if (afterCell.getCellType() == CellType_Constructor || afterCell.getCellType() == CellType_Injector) {
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
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);

        std::set<CellGenomeDescription> beforeGenomeRollout;
        rollout(beforeGenome, beforeGenomeRollout);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);

        return beforeGenomeRollout == afterGenomeRollout;
    }

    bool compareCellColorMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after, std::set<int> const& allowedColors)
    {
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            afterCellClone.color = 0;
            afterCellClone.cellTypeData = beforeCellClone.cellTypeData;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (!allowedColors.contains(afterCell.color)) {
                return false;
            }
            if (beforeCell.getCellType() == CellType_Constructor || beforeCell.getCellType() == CellType_Injector) {
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
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        std::optional<int> uniformColor;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            afterCellClone.color = 0;
            afterCellClone.cellTypeData = beforeCellClone.cellTypeData;
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
            if (beforeCell.getCellType() == CellType_Constructor || beforeCell.getCellType() == CellType_Injector) {
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
        auto beforeGenome = GenomeDescriptionService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(after);
        if (afterGenome.header != beforeGenome.header) {
            return false;
        }

        int uniformColor = allowedColor ? *allowedColor : afterGenome.cells.at(0).color;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome.cells, afterGenome.cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone.color = 0;
            afterCellClone.color = 0;
            afterCellClone.cellTypeData = beforeCellClone.cellTypeData;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (afterCell.color != uniformColor) {
                return false;
            }
            uniformColor = afterCell.color;
            if (beforeCell.getCellType() == CellType_Constructor || beforeCell.getCellType() == CellType_Injector) {
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
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Properties);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(comparePropertiesMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, neuronDataMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::NeuronData);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareNeuronDataMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, geometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Geometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareGeometryMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, individualGeometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CustomGeometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareIndividualGeometryMutation(genome, actualConstructor.genome));
    EXPECT_EQ(byteIndex, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, cellTypeMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(3))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellType);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareCellTypeMutation(genome, actualConstructor.genome));
    EXPECT_EQ(3, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, insertMutation_emptyGenome)
{
    auto cellColor = 3;
    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription()).setColor(cellColor)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Insertion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);

    auto actualGenomeDescription = GenomeDescriptionService::get().convertBytesToDescription(actualConstructor.genome);
    EXPECT_EQ(1, actualGenomeDescription.cells.size());
    EXPECT_EQ(cellColor, actualGenomeDescription.cells.front().color);
}

TEST_F(MutationTests, insertMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells({CellDescription()
                                                .setId(1)
                                                .setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))
                                                
                                                .setColor(genomeCellColors[0])});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Insertion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseSmallGenome)
{
    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
        CellGenomeDescription().setCellTypeData(BaseGenomeDescription()),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Deletion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_EQ(GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor.genome.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_preserveSelfReplication)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    auto afterGenome = GenomeDescriptionService::get().convertBytesToDescription(actualConstructor.genome);

    std::set<CellGenomeDescription> afterGenomeRollout;
    rollout(afterGenome, afterGenomeRollout);
    for (auto const& cell : afterGenomeRollout) {
        auto cellType = cell.getCellType();
        EXPECT_TRUE(cellType == CellType_Constructor || cellType == CellType_Injector);
    }
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_changeSelfReplication)
{
    _parameters.cellCopyMutationSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);

    EXPECT_EQ(GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor.genome.size());
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_partiallyEraseGenome)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareDeleteMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_selfReplicatorWithGenomeBelowMinSize)
{
    _parameters.features.customizeDeletionMutations = true;
    _parameters.cellCopyMutationDeletionMinSize = 3;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells(
        {CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()),
        CellGenomeDescription(),
        CellGenomeDescription(),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome)).setLivingState(LivingState_Activating)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actualConstructor.genome);
    
    EXPECT_EQ(3, actualGenome.cells.size());
}

TEST_F(MutationTests, deleteMutation_selfReplicatorWithGenomeAboveMinSize)
{
    _parameters.features.customizeDeletionMutations = true;
    _parameters.cellCopyMutationDeletionMinSize = 1;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescriptionService::get().convertDescriptionToBytes(GenomeDescription().setCells({
        CellGenomeDescription().setCellTypeData(ConstructorGenomeDescription().setMakeSelfCopy()),
        CellGenomeDescription(),
        CellGenomeDescription(),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome)).setLivingState(LivingState_Activating)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    auto actualGenome = GenomeDescriptionService::get().convertBytesToDescription(actualConstructor.genome);

    EXPECT_EQ(1, actualGenome.cells.size());
}

TEST_F(MutationTests, duplicateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Duplication);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor.genome));
    EXPECT_EQ(0, actualConstructor.genomeCurrentNodeIndex);
}

TEST_F(MutationTests, translateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Translation);
    }
    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);
    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareTranslateMutation(genome, actualConstructor.genome));
}

TEST_F(MutationTests, cellColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellCopyMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellCopyMutationColorTransitions[0][3] = true;
    _parameters.cellCopyMutationColorTransitions[0][5] = true;
    _parameters.cellCopyMutationColorTransitions[4][2] = true;
    _parameters.cellCopyMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColorPerSubgenome();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareCellColorMutation(genome, actualConstructor.genome, {1, 2, 4, 5}));
}

TEST_F(MutationTests, subgenomeColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellCopyMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellCopyMutationColorTransitions[0][3] = true;
    _parameters.cellCopyMutationColorTransitions[0][5] = true;
    _parameters.cellCopyMutationColorTransitions[4][2] = true;
    _parameters.cellCopyMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColorPerSubgenome();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::SubgenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareSubgenomeColorMutation(genome, actualConstructor.genome, {1, 2, 4, 5}));
}

TEST_F(MutationTests, genomeColorMutation)
{
    for (int i = 0; i < MAX_COLORS; ++i) {
        for (int j = 0; j < MAX_COLORS; ++j) {
            _parameters.cellCopyMutationColorTransitions[i][j] = false;
        }
    }
    _parameters.cellCopyMutationColorTransitions[0][3] = true;
    _parameters.cellCopyMutationColorTransitions[0][5] = true;
    _parameters.cellCopyMutationColorTransitions[4][2] = true;
    _parameters.cellCopyMutationColorTransitions[4][5] = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithUniformColor();

    auto data = DataDescription().addCells(
        {CellDescription().setId(1).setCellTypeData(ConstructorDescription().setGenome(genome).setGenomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::GenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1).cellTypeData);
    EXPECT_TRUE(compareGenomeColorMutation(genome, actualConstructor.genome, std::nullopt));
}
