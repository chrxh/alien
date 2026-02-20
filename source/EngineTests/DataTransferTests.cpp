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
        data.objects(
            {_descTestDataFactory->createNonDefaultObjectDesc(objectParameter),
             _descTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }

    _simulationFacade->setSimulationData(data);
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
        data.objects(
            {_descTestDataFactory->createNonDefaultObjectDesc(objectParameter),
             _descTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }

    _simulationFacade->setPreviewData(data);
    auto actualData = _simulationFacade->getPreviewData();

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
    auto actualData = _simulationFacade->getPreviewData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, multipleCells_genome_multipleGenes_multipleNodes)
{
    auto hexagon = DescEditService::get().createHex(DescEditService::CreateHexParameters().center({100.0f, 100.0f}).objectType(CellDesc()));


    auto data = Desc().addCreature(hexagon._objects, CreatureDesc(), GenomeDesc().genes({
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()}),
            GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc(), NodeDesc()}),
        }));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

TEST_F(DataTransferTests, setSimulationData_keepIdsStable)
{
    auto data = Desc()
                    .objects({ObjectDesc().id(0).type(StructureDesc()), ObjectDesc().id(1).type(FreeCellDesc())})
                    .energies({EnergyDesc().id(2), EnergyDesc().id(3)});
    data.addCreature({ObjectDesc().id(5)}, CreatureDesc().id(4), GenomeDesc());
    data.addCreature({ObjectDesc().id(6)}, CreatureDesc().id(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->setSimulationData(data);

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> expectedCellIds;
    for (auto const& object : data._objects) { expectedCellIds.insert(object._id); }
    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& object : actualData._objects) { actualCellIds.insert(object._id); }
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
    auto data = Desc()
                    .objects({ObjectDesc().id(0).type(FreeCellDesc()), ObjectDesc().id(1).type(FreeCellDesc())})
                    .energies({EnergyDesc().id(2), EnergyDesc().id(3)});
    data.addCreature({ObjectDesc().id(5)}, CreatureDesc().id(4), GenomeDesc());
    data.addCreature({ObjectDesc().id(6)}, CreatureDesc().id(5), GenomeDesc());

    _simulationFacade->setSimulationData(data);
    _simulationFacade->addAndSelectSimulationData(std::move(data));

    auto actualData = _simulationFacade->getSimulationData();

    std::unordered_set<uint64_t> actualCellIds;
    for (auto const& object : actualData._objects) { actualCellIds.insert(object._id); }
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

TEST_F(DataTransferTests, changeGenome_successful)
{
    auto const CreatureId = 1;

    auto data = Desc().addCreature({ObjectDesc()}, CreatureDesc().id(CreatureId), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->changeCreature(CreatureId, newGenome);
    ASSERT_TRUE(result);

    auto actualData = _simulationFacade->getSimulationData();

    ASSERT_EQ(0, actualData.getNumObjectsWithoutCreature());
    ASSERT_EQ(1, actualData._creatures.size());

    auto creature = actualData._creatures.front();
    ASSERT_EQ(1, actualData.getObjectsForCreature(creature._id).size());
    EXPECT_EQ(CreatureId, creature._id);

    // Find the genome for this creature
    auto genomeIt = std::find_if(actualData._genomes.begin(), actualData._genomes.end(), [&creature](auto const& g) { return g._id == creature._genomeId; });
    ASSERT_NE(genomeIt, actualData._genomes.end());
    ASSERT_EQ(1, genomeIt->_genes.size());

    auto gene = genomeIt->_genes.front();
    EXPECT_EQ(2, gene._nodes.size());
}

TEST_F(DataTransferTests, changeGenome_failed)
{
    auto constexpr CreatureId = 1;
    auto constexpr WrongCreatureId = 2;

    auto data = Desc().addCreature({ObjectDesc()}, CreatureDesc().id(CreatureId), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto newGenome = GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()})});
    auto result = _simulationFacade->changeCreature(WrongCreatureId, newGenome);
    ASSERT_FALSE(result);
}

TEST_F(DataTransferTests, getGenomeOfCreature_successful)
{
    auto constexpr CreatureId = 1;

    auto genome =
        GenomeDesc().name("Test Genome").genes({GeneDesc().name("Gene1").separation(true).nodes({NodeDesc(), NodeDesc()})});
    auto data = Desc().addCreature({ObjectDesc(), ObjectDesc(), ObjectDesc(), ObjectDesc()}, CreatureDesc().id(CreatureId), genome);

    _simulationFacade->setSimulationData(data);

    auto retrievedGenome = _simulationFacade->getGenomeOfCreature(CreatureId);

    ASSERT_TRUE(retrievedGenome.has_value());
    EXPECT_EQ("Test Genome", retrievedGenome->_name);
    ASSERT_EQ(1, retrievedGenome->_genes.size());
    EXPECT_EQ("Gene1", retrievedGenome->_genes.front()._name);
    EXPECT_TRUE(retrievedGenome->_genes.front()._separation);
    EXPECT_EQ(2, retrievedGenome->_genes.front()._nodes.size());
}

TEST_F(DataTransferTests, getGenomeOfCreature_nonExistentCreature)
{
    auto constexpr CreatureId = 1;
    auto constexpr WrongCreatureId = 2;

    auto data = Desc().addCreature({ObjectDesc()}, CreatureDesc().id(CreatureId), GenomeDesc());

    _simulationFacade->setSimulationData(data);

    auto result = _simulationFacade->getGenomeOfCreature(WrongCreatureId);
    EXPECT_FALSE(result.has_value());
}

TEST_F(DataTransferTests, getInspectedSimulationData)
{
    auto constexpr CreatureId1 = 1;
    auto constexpr CreatureId2 = 2;
    auto genome = GenomeDesc().genes({GeneDesc().separation(true).nodes({NodeDesc(), NodeDesc()})});
    auto genome2 = GenomeDesc();

    Desc data;
    data.addCreature({ObjectDesc().id(1), ObjectDesc().id(2)}, CreatureDesc().id(CreatureId1), genome);
    data.addCreature({ObjectDesc().id(3)}, CreatureDesc().id(CreatureId2), genome2);

    data.addConnection(1, 2);
    data.addConnection(2, 3);
    _simulationFacade->setSimulationData(data);

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
    auto data = Desc().objects({ObjectDesc().id(HighId).type(StructureDesc())});
    _simulationFacade->setSimulationData(data);

    NumberGenerator::get().setIds({1});
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
    auto actualData = _simulationFacade->getSimulationData();

    auto newId = NumberGenerator::get().createEntityId();
    EXPECT_TRUE(newId > HighId);
}
