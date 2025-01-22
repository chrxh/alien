#include <cmath>

#include <algorithm>
#include <ranges>
#include <cstdlib>
#include <boost/range/combine.hpp>

#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"

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
        std::vector<uint8_t> subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 14; ++i) {
            subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
                CellGenomeDescription().cellType(BaseGenomeDescription()).color(genomeCellColors[0]),
                CellGenomeDescription().cellType(DepotGenomeDescription()).color(genomeCellColors[1]),
                CellGenomeDescription().color(genomeCellColors[2]),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(genomeCellColors[2]),
                CellGenomeDescription()
                    .cellType(ConstructorGenomeDescription().genome(subGenome).mode(std::rand() % 100))
                    .color(genomeCellColors[0]),
            }));
        }
        return GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
            CellGenomeDescription().cellType(BaseGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(DepotGenomeDescription()).color(genomeCellColors[1]),
            CellGenomeDescription().color(genomeCellColors[0]),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(genomeCellColors[1]),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(SensorGenomeDescription()).color(genomeCellColors[2]),
            CellGenomeDescription().cellType(OscillatorGenomeDescription()).color(genomeCellColors[1]),
            CellGenomeDescription().cellType(AttackerGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(InjectorGenomeDescription().genome(subGenome)).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(MuscleGenomeDescription()).color(genomeCellColors[2]),
            CellGenomeDescription().cellType(DefenderGenomeDescription()).color(genomeCellColors[2]),
            CellGenomeDescription().cellType(ReconnectorGenomeDescription()).color(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColorPerSubgenome() const
    {
        std::vector<uint8_t> subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            auto color = genomeCellColors[i % genomeCellColors.size()];
            subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
                CellGenomeDescription().cellType(BaseGenomeDescription()).color(color),
                CellGenomeDescription().cellType(DepotGenomeDescription()).color(color),
                CellGenomeDescription().color(color),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(color),
                CellGenomeDescription()
                    .cellType(ConstructorGenomeDescription().genome(subGenome).mode(std::rand() % 100))
                    .color(color),
            }));
        };
        return GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
            CellGenomeDescription().cellType(BaseGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(DepotGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().color(genomeCellColors[0]),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(SensorGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(OscillatorGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(AttackerGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(InjectorGenomeDescription().genome(subGenome)).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(MuscleGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(DefenderGenomeDescription()).color(genomeCellColors[0]),
            CellGenomeDescription().cellType(ReconnectorGenomeDescription()).color(genomeCellColors[0]),
        }));
    }

    std::vector<uint8_t> createGenomeWithUniformColor() const
    {
        auto color = genomeCellColors[0];
        std::vector<uint8_t> subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription());
        for (int i = 0; i < 15; ++i) {
            subGenome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
                CellGenomeDescription().cellType(BaseGenomeDescription()).color(color),
                CellGenomeDescription().cellType(DepotGenomeDescription()).color(color),
                CellGenomeDescription().color(color),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(color),
                CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome).mode(std::rand() % 100)).color(color),
            }));
        };
        return GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
            CellGenomeDescription().cellType(BaseGenomeDescription()).color(color),
            CellGenomeDescription().cellType(DepotGenomeDescription()).color(color),
            CellGenomeDescription().color(color),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()).color(color),
            CellGenomeDescription().cellType(ConstructorGenomeDescription().genome(subGenome)).color(color),
            CellGenomeDescription().cellType(SensorGenomeDescription()).color(color),
            CellGenomeDescription().cellType(OscillatorGenomeDescription()).color(color),
            CellGenomeDescription().cellType(AttackerGenomeDescription()).color(color),
            CellGenomeDescription().cellType(InjectorGenomeDescription().genome(subGenome)).color(color),
            CellGenomeDescription().cellType(MuscleGenomeDescription()).color(color),
            CellGenomeDescription().cellType(DefenderGenomeDescription()).color(color),
            CellGenomeDescription().cellType(ReconnectorGenomeDescription()).color(color),
        }));
    }

    void rollout(GenomeDescription const& input, std::set<CellGenomeDescription>& result)
    {
        for (auto const& cell : input._cells) {
            if (auto subGenomeBytes = cell.getGenome()) {
                auto subGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(*subGenomeBytes);
                rollout(subGenome, result);
                auto cellClone = cell;
                cellClone.genome({});
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
        auto expectedGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actual);
        if (expectedGenome._header != actualGenome._header) {
            return false;
        }
        if (expectedGenome._cells.size() != actualGenome._cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome._cells, actualGenome._cells)) {
            if (expectedCell.getCellType() != actualCell.getCellType()) {
                return false;
            }
            if (expectedCell._color != actualCell._color) {
                return false;
            }
            if (expectedCell._referenceAngle != actualCell._referenceAngle) {
                return false;
            }
            if (expectedCell._numRequiredAdditionalConnections != actualCell._numRequiredAdditionalConnections) {
                return false;
            }
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell._cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell._cellTypeData);
                if (expectedConstructor._constructionAngle1 != actualConstructor._constructionAngle1) {
                    return false;
                }
                if (expectedConstructor._constructionAngle2 != actualConstructor._constructionAngle2) {
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
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell._cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell._cellTypeData);
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
        auto expectedGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actual);
        if (expectedGenome._header != actualGenome._header) {
            return false;
        }
        if (expectedGenome._cells.size() != actualGenome._cells.size()) {
            return false;
        }

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome._cells, actualGenome._cells)) {
            if (expectedCell.getCellType() != actualCell.getCellType()) {
                return false;
            }
            if (expectedCell.getCellType() != CellType_Base && expectedCell.getCellType() != CellType_Constructor
                && expectedCell.getCellType() != CellType_Injector && expectedCell != actualCell) {
                return false;
            }
            if (expectedCell._color != actualCell._color) {
                return false;
            }
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell._cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell._cellTypeData);
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
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell._cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell._cellTypeData);
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
        auto expectedGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actual);
        if (expectedGenome._cells.size() != actualGenome._cells.size()) {
            return false;
        }

        auto createCompareClone = [](CellGenomeDescription const& cell) {
            auto clone = cell;
            clone._referenceAngle = 0;
            clone._numRequiredAdditionalConnections = 0;
            if (clone.getCellType() == CellType_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(clone._cellTypeData);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor._genome = {};
                }
            }
            if (clone.getCellType() == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(clone._cellTypeData);
                if (!injector.isMakeGenomeCopy()) {
                    injector._genome = {};
                }
            }
            return clone;
        };

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome._cells, actualGenome._cells)) {
            if (createCompareClone(expectedCell) != createCompareClone(actualCell)) {
                return false;
            }
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell._cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell._cellTypeData);
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
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell._cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell._cellTypeData);
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
        auto expectedGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actual);
        expectedGenome._header._shape = ConstructionShape_Custom; //compare all expect shape
        actualGenome._header._shape = ConstructionShape_Custom;
        if (expectedGenome._header != actualGenome._header) {
            return false;
        }
        if (expectedGenome._cells.size() != actualGenome._cells.size()) {
            return false;
        }

        auto createCompareClone = [](CellGenomeDescription const& cell) {
            auto clone = cell;
            clone._referenceAngle = 0;
            clone._numRequiredAdditionalConnections = 0;
            if (clone.getCellType() == CellType_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(clone._cellTypeData);
                if (!constructor.isMakeGenomeCopy()) {
                    constructor._genome = {};
                }
                constructor._constructionAngle1 = 0;
                constructor._constructionAngle2 = 0;
            }
            if (clone.getCellType() == CellType_Injector) {
                auto& injector = std::get<InjectorGenomeDescription>(clone._cellTypeData);
                if (!injector.isMakeGenomeCopy()) {
                    injector._genome = {};
                }
            }
            return clone;
        };

        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome._cells, actualGenome._cells)) {
            if (createCompareClone(expectedCell) != createCompareClone(actualCell)) {
                return false;
            }
            if (expectedCell.getCellType() == CellType_Constructor) {
                auto expectedConstructor = std::get<ConstructorGenomeDescription>(expectedCell._cellTypeData);
                auto actualConstructor = std::get<ConstructorGenomeDescription>(actualCell._cellTypeData);
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
                auto expectedInjector = std::get<InjectorGenomeDescription>(expectedCell._cellTypeData);
                auto actualInjector = std::get<InjectorGenomeDescription>(actualCell._cellTypeData);
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
        auto expectedGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(expected);
        auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actual);
        if (expectedGenome._header != actualGenome._header) {
            return false;
        }
        if (expectedGenome._cells.size() != actualGenome._cells.size()) {
            return false;
        }
        for (auto const& [expectedCell, actualCell] : boost::combine(expectedGenome._cells, actualGenome._cells)) {
            if (std::abs(expectedCell._referenceAngle - actualCell._referenceAngle) > NEAR_ZERO) {
                return false;
            }
            if (std::abs(expectedCell._energy- actualCell._energy) > NEAR_ZERO) {
                return false;
            }
            if (expectedCell._color != actualCell._color) {
                return false;
            }
            if (expectedCell._numRequiredAdditionalConnections != actualCell._numRequiredAdditionalConnections) {
                return false;
            }
        }
        return true;
    }

    bool compareInsertMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after)
    {
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);
        if (afterGenome._header != beforeGenome._header) {
            return false;
        }
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(genomeCellColors, cell._color) == genomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& beforeCell : beforeGenome._cells) {
            auto matchingAfterCells = afterGenome._cells | std::views::filter([&beforeCell](auto const& afterCell) {
                                          auto beforeCellClone = beforeCell;
                                          auto afterCellClone = afterCell;
                                          afterCellClone._cellTypeData = beforeCellClone._cellTypeData;
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
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);
        if (afterGenome._header != beforeGenome._header) {
            return false;
        }
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);
        for (auto const& cell : afterGenomeRollout) {
            if (std::ranges::find(genomeCellColors, cell._color) == genomeCellColors.end()) {
                return false;
            }
        }
        for (auto const& afterCell : afterGenome._cells) {
            auto matchingBeforeCells = beforeGenome._cells | std::views::filter([&afterCell](auto const& beforeCell) {
                                           auto beforeCellClone = beforeCell;
                                           auto afterCellClone = afterCell;
                                           afterCellClone._cellTypeData = beforeCellClone._cellTypeData;
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
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);

        std::set<CellGenomeDescription> beforeGenomeRollout;
        rollout(beforeGenome, beforeGenomeRollout);
        std::set<CellGenomeDescription> afterGenomeRollout;
        rollout(afterGenome, afterGenomeRollout);

        return beforeGenomeRollout == afterGenomeRollout;
    }

    bool compareCellColorMutation(std::vector<uint8_t> const& before, std::vector<uint8_t> const& after, std::set<int> const& allowedColors)
    {
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);
        if (afterGenome._header != beforeGenome._header) {
            return false;
        }

        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome._cells, afterGenome._cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone._color = 0;
            afterCellClone._color = 0;
            afterCellClone._cellTypeData = beforeCellClone._cellTypeData;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (!allowedColors.contains(afterCell._color)) {
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
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);
        if (afterGenome._header != beforeGenome._header) {
            return false;
        }

        std::optional<int> uniformColor;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome._cells, afterGenome._cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone._color = 0;
            afterCellClone._color = 0;
            afterCellClone._cellTypeData = beforeCellClone._cellTypeData;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (!allowedColors.contains(afterCell._color)) {
                return false;
            }
            if (uniformColor && afterCell._color != *uniformColor) {
                return false;
            }
            uniformColor = afterCell._color;
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
        auto beforeGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(before);
        auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(after);
        if (afterGenome._header != beforeGenome._header) {
            return false;
        }

        int uniformColor = allowedColor ? *allowedColor : afterGenome._cells.at(0)._color;
        for (auto const& [beforeCell, afterCell] : boost::combine(beforeGenome._cells, afterGenome._cells)) {

            auto beforeCellClone = beforeCell;
            auto afterCellClone = afterCell;
            beforeCellClone._color = 0;
            afterCellClone._color = 0;
            afterCellClone._cellTypeData = beforeCellClone._cellTypeData;
            if (beforeCellClone != afterCellClone) {
                return false;
            }
            if (afterCell._color != uniformColor) {
                return false;
            }
            uniformColor = afterCell._color;
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
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Properties);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(comparePropertiesMutation(genome, actualConstructor._genome));
    EXPECT_EQ(byteIndex, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, neuronDataMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::NeuronData);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareNeuronDataMutation(genome, actualConstructor._genome));
    EXPECT_EQ(byteIndex, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, geometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Geometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareGeometryMutation(genome, actualConstructor._genome));
    EXPECT_EQ(byteIndex, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, individualGeometryMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();
    int byteIndex = 0;

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(byteIndex))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CustomGeometry);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareIndividualGeometryMutation(genome, actualConstructor._genome));
    EXPECT_EQ(byteIndex, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, cellTypeMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(3))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellType);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareCellTypeMutation(genome, actualConstructor._genome));
    EXPECT_EQ(3, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, insertMutation_emptyGenome)
{
    auto cellColor = 3;
    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription()).color(cellColor)});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Insertion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);

    auto actualGenomeDescription = GenomeDescriptionConverterService::get().convertBytesToDescription(actualConstructor._genome);
    EXPECT_EQ(1, actualGenomeDescription._cells.size());
    EXPECT_EQ(cellColor, actualGenomeDescription._cells.front()._color);
}

TEST_F(MutationTests, insertMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells({CellDescription()
                                                .id(1)
                                                .cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))
                                                
                                                .color(genomeCellColors[0])});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Insertion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor._genome));
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseSmallGenome)
{
    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
        CellGenomeDescription().cellType(BaseGenomeDescription()),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1, MutationType::Deletion);

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_EQ(GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor._genome.size());
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_preserveSelfReplication)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    auto afterGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actualConstructor._genome);

    std::set<CellGenomeDescription> afterGenomeRollout;
    rollout(afterGenome, afterGenomeRollout);
    for (auto const& cell : afterGenomeRollout) {
        auto cellType = cell.getCellType();
        EXPECT_TRUE(cellType == CellType_Constructor || cellType == CellType_Injector);
    }
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_eraseLargeGenome_changeSelfReplication)
{
    _parameters.cellCopyMutationSelfReplication = true;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);

    EXPECT_EQ(GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription()).size(), actualConstructor._genome.size());
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_partiallyEraseGenome)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareDeleteMutation(genome, actualConstructor._genome));
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, deleteMutation_selfReplicatorWithGenomeBelowMinSize)
{
    _parameters.features.customizeDeletionMutations = true;
    _parameters.cellCopyMutationDeletionMinSize = 3;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells(
        {CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
        CellGenomeDescription(),
        CellGenomeDescription(),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome)).livingState(LivingState_Activating)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actualConstructor._genome);
    
    EXPECT_EQ(3, actualGenome._cells.size());
}

TEST_F(MutationTests, deleteMutation_selfReplicatorWithGenomeAboveMinSize)
{
    _parameters.features.customizeDeletionMutations = true;
    _parameters.cellCopyMutationDeletionMinSize = 1;
    _simulationFacade->setSimulationParameters(_parameters);

    auto genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeDescription().cells({
        CellGenomeDescription().cellType(ConstructorGenomeDescription().makeSelfCopy()),
        CellGenomeDescription(),
        CellGenomeDescription(),
    }));

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome)).livingState(LivingState_Activating)});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Deletion);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    auto actualGenome = GenomeDescriptionConverterService::get().convertBytesToDescription(actualConstructor._genome);

    EXPECT_EQ(1, actualGenome._cells.size());
}

TEST_F(MutationTests, duplicateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Duplication);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareInsertMutation(genome, actualConstructor._genome));
    EXPECT_EQ(0, actualConstructor._genomeCurrentNodeIndex);
}

TEST_F(MutationTests, translateMutation)
{
    auto genome = createGenomeWithMultipleCellsWithDifferentFunctions();

    auto data = DataDescription().addCells(
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::Translation);
    }
    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);
    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareTranslateMutation(genome, actualConstructor._genome));
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
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::CellColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareCellColorMutation(genome, actualConstructor._genome, {1, 2, 4, 5}));
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
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::SubgenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareSubgenomeColorMutation(genome, actualConstructor._genome, {1, 2, 4, 5}));
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
        {CellDescription().id(1).cellType(ConstructorDescription().genome(genome).genomeCurrentNodeIndex(0))});

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 10000; ++i) {
        _simulationFacade->testOnly_mutate(1, MutationType::GenomeColor);
    }

    auto actualData = _simulationFacade->getSimulationData();
    auto actualCellById = getCellById(actualData);

    auto actualConstructor = std::get<ConstructorDescription>(actualCellById.at(1)._cellTypeData);
    EXPECT_TRUE(compareGenomeColorMutation(genome, actualConstructor._genome, std::nullopt));
}
