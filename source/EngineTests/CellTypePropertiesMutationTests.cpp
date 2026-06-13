#include <optional>
#include <type_traits>
#include <variant>

#include <gtest/gtest.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationFacade.h>

#include "MutationTestsBase.h"

class CellTypePropertiesMutationTests : public MutationTestsBase
{
protected:
    template <typename T>
    struct IsVariant : std::false_type
    {};
    template <typename... Ts>
    struct IsVariant<std::variant<Ts...>> : std::true_type
    {};

    // Resets all mutable cell type properties to their defaults while preserving the active cell type and its active mode.
    static void resetCellTypeProperties(CellTypeGenomeDesc& cellType)
    {
        std::visit(
            [](auto& cellTypeData) {
                using CellTypeData = std::decay_t<decltype(cellTypeData)>;
                if constexpr (requires { cellTypeData._mode; }) {
                    using ModeType = std::decay_t<decltype(cellTypeData._mode)>;
                    if constexpr (IsVariant<ModeType>::value) {
                        auto mode = cellTypeData._mode;
                        std::visit([](auto& modeData) { modeData = std::decay_t<decltype(modeData)>{}; }, mode);
                        cellTypeData = CellTypeData{};
                        cellTypeData._mode = mode;
                        return;
                    }
                }
                cellTypeData = CellTypeData{};
            },
            cellType);
    }

    bool compareAllExceptCellTypeProperties(GenomeDesc expected, GenomeDesc actual)
    {
        auto reset = [](GenomeDesc& genome) {
            genome._lineageId = 0;
            genome._prevLineageId = std::nullopt;
            genome._accumulatedMutations = 0.0f;
            for (auto& gene : genome._genes) {
                for (auto& node : gene._nodes) {
                    resetCellTypeProperties(node._cellType);
                }
            }
        };
        reset(expected);
        reset(actual);
        return expected == actual;
    }

    template <typename Predicate>
    std::optional<NodeDesc> findNode(GenomeDesc const& genome, Predicate&& predicate) const
    {
        for (auto const& gene : genome._genes) {
            for (auto const& node : gene._nodes) {
                if (predicate(node)) {
                    return node;
                }
            }
        }
        return std::nullopt;
    }
};

TEST_F(CellTypePropertiesMutationTests, cellTypePropertiesMutation_changesScalarBoolAndEnumProperties)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto originalDigestor = findNode(genome, [](NodeDesc const& node) { return std::holds_alternative<DigestorGenomeDesc>(node._cellType); });
    auto actualDigestor = findNode(actualGenome, [](NodeDesc const& node) { return std::holds_alternative<DigestorGenomeDesc>(node._cellType); });
    ASSERT_TRUE(originalDigestor);
    ASSERT_TRUE(actualDigestor);
    EXPECT_NE(
        std::get<DigestorGenomeDesc>(originalDigestor->_cellType)._rawEnergyConductivity,
        std::get<DigestorGenomeDesc>(actualDigestor->_cellType)._rawEnergyConductivity);

    auto originalMemory = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<MemoryGenomeDesc>(node._cellType)
            && std::holds_alternative<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(node._cellType)._mode);
    });
    auto actualMemory = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<MemoryGenomeDesc>(node._cellType)
            && std::holds_alternative<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalMemory);
    ASSERT_TRUE(actualMemory);
    EXPECT_NE(
        std::get<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(originalMemory->_cellType)._mode)._readOnly,
        std::get<SignalStorageGenomeDesc>(std::get<MemoryGenomeDesc>(actualMemory->_cellType)._mode)._readOnly);

    auto originalCommunicator = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    auto actualCommunicator = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalCommunicator);
    ASSERT_TRUE(actualCommunicator);
    EXPECT_NE(
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(originalCommunicator->_cellType)._mode)._restrictToLineage,
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(actualCommunicator->_cellType)._mode)._restrictToLineage);
}

TEST_F(CellTypePropertiesMutationTests, cellTypePropertiesMutation_changesBitsetPropertiesBitwise)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_mutate(1);

    auto actualGenome = getMutatedGenome();

    auto originalSensor = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<SensorGenomeDesc>(node._cellType)
            && std::holds_alternative<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(node._cellType)._mode);
    });
    auto actualSensor = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<SensorGenomeDesc>(node._cellType)
            && std::holds_alternative<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalSensor);
    ASSERT_TRUE(actualSensor);
    EXPECT_EQ(
        (std::get<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(originalSensor->_cellType)._mode)._restrictToColors ^ Const::RestrictToColors_Max),
        std::get<DetectFreeCellGenomeDesc>(std::get<SensorGenomeDesc>(actualSensor->_cellType)._mode)._restrictToColors);

    auto originalMemory = findNode(genome, [](NodeDesc const& node) { return std::holds_alternative<MemoryGenomeDesc>(node._cellType); });
    auto actualMemory = findNode(actualGenome, [](NodeDesc const& node) { return std::holds_alternative<MemoryGenomeDesc>(node._cellType); });
    ASSERT_TRUE(originalMemory);
    ASSERT_TRUE(actualMemory);
    EXPECT_EQ(
        static_cast<uint16_t>(std::get<MemoryGenomeDesc>(originalMemory->_cellType)._channelBitMask ^ Const::MemoryChannelBitMask_Max),
        std::get<MemoryGenomeDesc>(actualMemory->_cellType)._channelBitMask);

    auto originalCommunicator = findNode(genome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    auto actualCommunicator = findNode(actualGenome, [](NodeDesc const& node) {
        return std::holds_alternative<CommunicatorGenomeDesc>(node._cellType)
            && std::holds_alternative<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(node._cellType)._mode);
    });
    ASSERT_TRUE(originalCommunicator);
    ASSERT_TRUE(actualCommunicator);
    EXPECT_EQ(
        (std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(originalCommunicator->_cellType)._mode)._restrictToColors ^ Const::RestrictToColors_Max),
        std::get<ReceiverGenomeDesc>(std::get<CommunicatorGenomeDesc>(actualCommunicator->_cellType)._mode)._restrictToColors);
}

TEST_F(CellTypePropertiesMutationTests, cellTypePropertiesMutation_doesNotChangeOtherAttributes)
{
    auto genome = createTestGenome();
    genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);
    genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc().nodeProbability(1.0f).sigma(1.0f).discreteChangeProbability(1.0f);

    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc(), genome);

    _simulationFacade->setSimulationData(data);
    for (int i = 0; i < 100; ++i) {
        _simulationFacade->testOnly_mutate(1);
    }

    auto actualGenome = getMutatedGenome();

    EXPECT_TRUE(compareAllExceptCellTypeProperties(genome, actualGenome));
}
