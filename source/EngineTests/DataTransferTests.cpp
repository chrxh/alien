#include <algorithm>
#include <ranges>

#include <gtest/gtest.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/NumberGenerator.h>
#include <EngineInterface/SimulationFacade.h>

#include <EngineTestData/DescTestDataFactory.h>

#include "IntegrationTestFramework.h"


class DataTransferTests : public IntegrationTestFramework
{
public:
    DataTransferTests()
        : IntegrationTestFramework()
    {
        _descTestDataFactory = &DescTestDataFactory::get();
    }

protected:
    DescTestDataFactory* _descTestDataFactory;
};

using ObjectParameter = DescTestDataFactory::ObjectParameter;
class DataTransferTests_AllObjectTypes
    : public DataTransferTests
    , public testing::WithParamInterface<ObjectParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllObjectTypes,
    DataTransferTests_AllObjectTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllObjectParameters()));

TEST_P(DataTransferTests_AllObjectTypes, objectsWithEmptyGenomes)
{
    auto objectParameter = GetParam();

    Desc data;
    if (objectParameter.objectType == ObjectType_Cell) {
        data.addCreature({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
        data.addCreature({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
    } else {
        data.objects({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter), _descTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_P(DataTransferTests_AllObjectTypes, objectsWithEmptyGenomes_preview)
{
    auto objectParameter = GetParam();

    Desc data;
    if (objectParameter.objectType == ObjectType_Cell) {
        data.addCreature({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
        data.addCreature({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
    } else {
        data.objects({_descTestDataFactory->createNonDefaultObjectDesc(objectParameter), _descTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getPreviewData();

    // Preview mode resets lastUpdate to 0
    for (auto& object : data._objects) {
        if (object.getObjectType() == ObjectType_Cell) {
            object.getCellRef()._lastUpdate = 0;
        }
    }

    EXPECT_TRUE(compare(data, actualData));
}

using NodeParameter = DescTestDataFactory::NodeParameter;
class DataTransferTests_AllNodeTypes
    : public DataTransferTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllNodeTypes,
    DataTransferTests_AllNodeTypes,
    ::testing::ValuesIn(DescTestDataFactory::get().getAllNodeParameters()));

TEST_P(DataTransferTests_AllNodeTypes, objectsWithNonEmptyGenomes_oneNode)
{
    auto nodeParameter = GetParam();

    auto [creature1, genome1] = _descTestDataFactory->createNonDefaultCreatureDesc(nodeParameter);
    auto [creature2, genome2] = _descTestDataFactory->createNonDefaultCreatureDesc(nodeParameter);

    Desc data;
    data.addCreature({ObjectDesc()}, creature1, genome1);
    data.addCreature({ObjectDesc()}, creature2, genome2);

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_P(DataTransferTests_AllNodeTypes, objectsWithNonEmptyGenomes_oneNode_preview)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descTestDataFactory->createNonDefaultCreatureDesc(nodeParameter);

    Desc data;
    data.addCreature({ObjectDesc()}, creature, genome);

    _simulationFacade->setPreviewData(data);
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, multipleCells_genome_multipleGenes_multipleNodes)
{
    auto hexagon = DescEditService::get().createHex(DescEditService::CreateHexParameters().center({100.0f, 100.0f}).objectType(CellDesc()));


    auto data = Desc().addCreature(
        hexagon._objects,
        CreatureDesc(),
        GenomeDesc().genes({
            GeneDesc().nodes({NodeDesc(), NodeDesc()}),
            GeneDesc().nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, setSimulationData_keepIdsStable)
{
    auto data =
        Desc().objects({ObjectDesc().id(0).type(SolidDesc()), ObjectDesc().id(1).type(FreeCellDesc())}).energies({EnergyDesc().id(2), EnergyDesc().id(3)});
    data.addCreature({ObjectDesc().id(5)}, CreatureDesc().id(4), GenomeDesc());
    data.addCreature({ObjectDesc().id(6)}, CreatureDesc().id(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    for (auto const& object : data._objects) {
        expectedCellIds.insert(object._id);
    }
    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& object : actualData._objects) {
        actualCellIds.insert(object._id);
    }
    EXPECT_EQ(expectedCellIds, actualCellIds);

    std::unordered_set<uint64_t> expectedCreatureIds;
    for (auto const& creature : data._creatures) {
        expectedCreatureIds.insert(creature._id);
    }
    std::unordered_set<uint64_t> actualCreatureIds;
    for (auto const& creature : actualData._creatures) {
        actualCreatureIds.insert(creature._id);
    }
    EXPECT_EQ(expectedCreatureIds, actualCreatureIds);

    std::unordered_set<uint64_t> expectedParticleIds;
    for (auto const& energyParticle : data._energies) {
        expectedParticleIds.insert(energyParticle._id);
    }
    std::unordered_set<uint64_t> actualParticleIds;
    for (auto const& energyParticle : actualData._energies) {
        actualParticleIds.insert(energyParticle._id);
    }
    EXPECT_EQ(expectedParticleIds, actualParticleIds);
}

TEST_F(DataTransferTests, addAndSelectSimulationData_assignNewIds)
{
    auto data =
        Desc().objects({ObjectDesc().id(0).type(FreeCellDesc()), ObjectDesc().id(1).type(FreeCellDesc())}).energies({EnergyDesc().id(2), EnergyDesc().id(3)});
    data.addCreature({ObjectDesc().id(5)}, CreatureDesc().id(4), GenomeDesc());
    data.addCreature({ObjectDesc().id(6)}, CreatureDesc().id(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& object : actualData._objects) {
        actualCellIds.insert(object._id);
    }
    EXPECT_EQ(2 * 4, actualCellIds.size());

    std::unordered_set<uint64_t> actualCreatureIds;
    for (auto const& creature : actualData._creatures) {
        actualCreatureIds.insert(creature._id);
    }
    EXPECT_EQ(2 * 2, actualCreatureIds.size());

    std::unordered_set<uint64_t> actualParticleIds;
    for (auto const& energyParticle : actualData._energies) {
        actualParticleIds.insert(energyParticle._id);
    }
    EXPECT_EQ(2 * 2, actualParticleIds.size());
}

TEST_F(DataTransferTests, injectGenomeToSelectedCreatures_singleCreature)
{
    auto constexpr CreatureId = 1;

    auto data = Desc().addCreature({ObjectDesc().pos({100, 100})}, CreatureDesc().id(CreatureId), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({99, 99}, {101, 101});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    EXPECT_EQ(CreatureId, creature._id);

    auto genomeIt = std::find_if(actualData._genomes.begin(), actualData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
    ASSERT_NE(genomeIt, actualData._genomes.end());
    ASSERT_EQ(1, genomeIt->_genes.size());
    EXPECT_EQ(2, genomeIt->_genes.front()._nodes.size());
}

TEST_F(DataTransferTests, injectGenomeToSelectedCreatures_noSelection)
{
    auto constexpr CreatureId = 1;

    auto originalGenome = GenomeDesc();
    auto data = Desc().addCreature({ObjectDesc().pos({100, 100})}, CreatureDesc().id(CreatureId), originalGenome);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->removeSelection();

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(1, actualData._creatures.size());
    ASSERT_EQ(1, actualData._genomes.size());
    EXPECT_EQ(0, actualData._genomes.front()._genes.size());
}

TEST_F(DataTransferTests, injectGenomeToSelectedCreatures_multipleCreatures_onlySelectedAffected)
{
    auto constexpr CreatureId1 = 1;
    auto constexpr CreatureId2 = 2;

    auto originalGenome1 = GenomeDesc();
    auto originalGenome2 = GenomeDesc();

    auto data = Desc()
                    .addCreature({ObjectDesc().pos({100, 100})}, CreatureDesc().id(CreatureId1), originalGenome1)
                    .addCreature({ObjectDesc().pos({200, 200})}, CreatureDesc().id(CreatureId2), originalGenome2);

    _simulationFacade->setSimulationData(data);

    _simulationFacade->setSelection({99, 99}, {101, 101});

    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(2, actualData._creatures.size());

    for (auto const& creature : actualData._creatures) {
        auto genomeIt =
            std::find_if(actualData._genomes.begin(), actualData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
        ASSERT_NE(genomeIt, actualData._genomes.end());

        if (creature._id == CreatureId1) {
            ASSERT_EQ(1, genomeIt->_genes.size());
            EXPECT_EQ(2, genomeIt->_genes.front()._nodes.size());
        } else {
            EXPECT_EQ(0, genomeIt->_genes.size());
        }
    }
}

TEST_F(DataTransferTests, getInspectedSimulationData)
{
    auto constexpr CreatureId1 = 1;
    auto constexpr CreatureId2 = 2;
    auto genome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc(), NodeDesc()})});
    auto genome2 = GenomeDesc();

    Desc data;
    data.addCreature({ObjectDesc().id(1), ObjectDesc().id(2)}, CreatureDesc().id(CreatureId1), genome);
    data.addCreature({ObjectDesc().id(3)}, CreatureDesc().id(CreatureId2), genome2);

    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

    _simulationFacade->testOnly_zeroTransferData();
    auto inspectedData = _simulationFacade->getInspectedSimulationData({1, 2});
    ASSERT_EQ(2, inspectedData._creatures.size());

    auto creature = inspectedData.getCreatureRef(CreatureId1);

    EXPECT_EQ(2, inspectedData.getObjectsForCreature(creature._id).size());

    auto object1 = inspectedData.getObjectRef(1);
    auto object2 = inspectedData.getObjectRef(2);

    // Find the genome for this creature
    auto genomeIt =
        std::find_if(inspectedData._genomes.begin(), inspectedData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
    ASSERT_NE(genomeIt, inspectedData._genomes.end());
    EXPECT_EQ(genome, *genomeIt);
}

TEST_F(DataTransferTests, adaptIdGenerator_objects)
{
    auto constexpr HighId = 1000000;
    auto data = Desc().objects({ObjectDesc().id(HighId).type(SolidDesc())});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_energyParticles)
{
    auto constexpr HighId = 1000000;
    auto data = Desc().energies({EnergyDesc().id(HighId)});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_creatures)
{
    auto constexpr HighId = 1000000;
    auto data = Desc().addCreature({ObjectDesc().id(1)}, CreatureDesc().id(HighId));
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_genomes)
{
    auto constexpr HighId = 1000000;
    auto data = Desc().addCreature({ObjectDesc().id(2)}, CreatureDesc().id(1), GenomeDesc().id(HighId));
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
    _simulationFacade->testOnly_zeroTransferData();
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}

TEST_F(DataTransferTests, adaptIdGenerator_injectGenomeToSelectedCreatures)
{
    auto constexpr CreatureId = 1;
    auto constexpr HighId = 1000000;

    auto data = Desc().addCreature({ObjectDesc().pos({100, 100})}, CreatureDesc().id(CreatureId), GenomeDesc());
    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSelection({99, 99}, {101, 101});

    // Force the injected genome to receive a high id from the number generator
    NumberGenerator::get().setIds({HighId});
    auto newGenome = GenomeDesc().genes({GeneDesc().nodes({NodeDesc()})});
    _simulationFacade->injectGenomeToSelectedCreatures(newGenome);

    // Sync host number generator from GPU; the injected genome's id must have been registered on the GPU during injection
    _simulationFacade->testOnly_syncNumberGenerator();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}
