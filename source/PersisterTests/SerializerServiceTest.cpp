#include <iostream>
#include <gtest/gtest.h>

#include <EngineTestData/DescriptionTestDataFactory.h>

#include <PersisterInterface/SerializerService.h>


class SerializerServiceTests : public ::testing::Test
{
public:
    SerializerServiceTests()
    {
        _descriptionTestDataFactory = &DescriptionTestDataFactory::get();
        _serializerService = &SerializerService::get();
    }

    void testSerializationAndDeserialization(Desc const& data)
    {
        DeserializedSimulation deserializedSimulationBefore{.mainData = data};
        SerializedSimulation serializedSimulation;
        _serializerService->serializeSimulationToStrings(serializedSimulation, deserializedSimulationBefore);

        DeserializedSimulation deserializedSimulationAfter;
        _serializerService->deserializeSimulationFromStrings(deserializedSimulationAfter, serializedSimulation);

        // Debug printing
        auto& before = deserializedSimulationBefore.mainData;
        auto& after = deserializedSimulationAfter.mainData;
        
        std::sort(before._objects.begin(), before._objects.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        std::sort(after._objects.begin(), after._objects.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        std::sort(before._creatures.begin(), before._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        std::sort(after._creatures.begin(), after._creatures.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        std::sort(before._genomes.begin(), before._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        std::sort(after._genomes.begin(), after._genomes.end(), [](auto const& left, auto const& right) { return left._id < right._id; });
        
        if (before._objects != after._objects || before._creatures != after._creatures || before._genomes != after._genomes) {
            std::cout << "\n=== DEBUG: Serialization mismatch ===" << std::endl;
            std::cout << "  objects equal: " << (before._objects == after._objects ? "true" : "false") << std::endl;
            std::cout << "  creatures equal: " << (before._creatures == after._creatures ? "true" : "false") << std::endl;
            std::cout << "  genomes equal: " << (before._genomes == after._genomes ? "true" : "false") << std::endl;
            
            if (!before._creatures.empty() && !after._creatures.empty() && before._creatures[0] != after._creatures[0]) {
                auto& bc = before._creatures[0];
                auto& ac = after._creatures[0];
                std::cout << "  Creature comparison:" << std::endl;
                std::cout << "    id: " << bc._id << " vs " << ac._id << std::endl;
                std::cout << "    genomeId: " << bc._genomeId << " vs " << ac._genomeId << std::endl;
                std::cout << "    numObjects: " << bc._numObjects << " vs " << ac._numObjects << std::endl;
                std::cout << "    generation: " << bc._generation << " vs " << ac._generation << std::endl;
            }
            if (!before._genomes.empty() && !after._genomes.empty() && before._genomes[0] != after._genomes[0]) {
                auto& bg = before._genomes[0];
                auto& ag = after._genomes[0];
                std::cout << "  Genome comparison:" << std::endl;
                std::cout << "    id: " << bg._id << " vs " << ag._id << std::endl;
                std::cout << "    name: '" << bg._name << "' vs '" << ag._name << "'" << std::endl;
                std::cout << "    lineageId: " << bg._lineageId << " vs " << ag._lineageId << std::endl;
            }
            if (!before._objects.empty() && !after._objects.empty() && before._objects[0] != after._objects[0]) {
                auto& bo = before._objects[0];
                auto& ao = after._objects[0];
                std::cout << "  Object comparison:" << std::endl;
                std::cout << "    id: " << bo._id << " vs " << ao._id << std::endl;
                std::cout << "    type: " << bo.getObjectType() << " vs " << ao.getObjectType() << std::endl;
                std::cout << "    pos.x: " << bo._pos.x << " vs " << ao._pos.x << std::endl;
                std::cout << "    pos.y: " << bo._pos.y << " vs " << ao._pos.y << std::endl;
                std::cout << "    vel.x: " << bo._vel.x << " vs " << ao._vel.x << std::endl;
                std::cout << "    vel.y: " << bo._vel.y << " vs " << ao._vel.y << std::endl;
                std::cout << "    color: " << bo._color << " vs " << ao._color << std::endl;
                std::cout << "    fixed: " << bo._fixed << " vs " << ao._fixed << std::endl;
                std::cout << "    sticky: " << bo._sticky << " vs " << ao._sticky << std::endl;
                std::cout << "    stiffness: " << bo._stiffness << " vs " << ao._stiffness << std::endl;
                std::cout << "    connections.size: " << bo._connections.size() << " vs " << ao._connections.size() << std::endl;
                std::cout << "    type variant index: " << bo._type.index() << " vs " << ao._type.index() << std::endl;
                if (bo.getObjectType() == ObjectType_Cell && ao.getObjectType() == ObjectType_Cell) {
                    auto& bc = bo.getCellRef();
                    auto& ac = ao.getCellRef();
                    std::cout << "    Cell details:" << std::endl;
                    std::cout << "      creatureId: " << bc._creatureId << " vs " << ac._creatureId << std::endl;
                    std::cout << "      usableEnergy: " << bc._usableEnergy << " vs " << ac._usableEnergy << std::endl;
                    std::cout << "      rawEnergy: " << bc._rawEnergy << " vs " << ac._rawEnergy << std::endl;
                    std::cout << "      reservedEnergy: " << bc._reservedEnergy << " vs " << ac._reservedEnergy << std::endl;
                    std::cout << "      age: " << bc._age << " vs " << ac._age << std::endl;
                    std::cout << "      cellState: " << static_cast<int>(bc._cellState) << " vs " << static_cast<int>(ac._cellState) << std::endl;
                    std::cout << "      geneIndex: " << bc._geneIndex << " vs " << ac._geneIndex << std::endl;
                    std::cout << "      nodeIndex: " << bc._nodeIndex << " vs " << ac._nodeIndex << std::endl;
                    std::cout << "      parentNodeIndex: " << bc._parentNodeIndex << " vs " << ac._parentNodeIndex << std::endl;
                    std::cout << "      headCell: " << bc._headCell << " vs " << ac._headCell << std::endl;
                    std::cout << "      frontAngle has_value: " << bc._frontAngle.has_value() << " vs " << ac._frontAngle.has_value() << std::endl;
                    std::cout << "      frontAngleId: " << bc._frontAngleId << " vs " << ac._frontAngleId << std::endl;
                    std::cout << "      activationTime: " << bc._activationTime << " vs " << ac._activationTime << std::endl;
                    std::cout << "      event: " << static_cast<int>(bc._event) << " vs " << static_cast<int>(ac._event) << std::endl;
                    std::cout << "      eventCounter: " << bc._eventCounter << " vs " << ac._eventCounter << std::endl;
                    std::cout << "      constructor has_value: " << bc._constructor.has_value() << " vs " << ac._constructor.has_value() << std::endl;
                    std::cout << "      cellType variant index: " << bc._cellType.index() << " vs " << ac._cellType.index() << std::endl;
                    
                    // Compare neural networks
                    std::cout << "      NN weights equal: " << (bc._neuralNetwork._weights == ac._neuralNetwork._weights ? "true" : "false") << std::endl;
                    std::cout << "      NN biases equal: " << (bc._neuralNetwork._biases == ac._neuralNetwork._biases ? "true" : "false") << std::endl;
                    std::cout << "      NN activationFunctions equal: " << (bc._neuralNetwork._activationFunctions == ac._neuralNetwork._activationFunctions ? "true" : "false") << std::endl;
                    std::cout << "      NN connectionWeights equal: " << (bc._neuralNetwork._connectionWeights == ac._neuralNetwork._connectionWeights ? "true" : "false") << std::endl;
                    
                    // Compare signal
                    std::cout << "      signal channels equal: " << (bc._signal._channels == ac._signal._channels ? "true" : "false") << std::endl;
                    std::cout << "      signal numTimesSent: " << bc._signal._numTimesSent << " vs " << ac._signal._numTimesSent << std::endl;
                }
            }
            std::cout << "===================================" << std::endl;
        }

        EXPECT_TRUE(_descriptionTestDataFactory->compare(deserializedSimulationBefore.mainData, deserializedSimulationAfter.mainData));
    }

protected:
    DescriptionTestDataFactory* _descriptionTestDataFactory;
    SerializerService* _serializerService;
};

TEST_F(SerializerServiceTests, singleEnergyParticle)
{
    Desc data;
    data._energies.emplace_back(_descriptionTestDataFactory->createNonDefaultEnergyDesc());

    testSerializationAndDeserialization(data);
}

using ObjectParameter = DescriptionTestDataFactory::ObjectParameter;
class SerializerServiceTests_AllCellTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<ObjectParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllCellTypes,
    SerializerServiceTests_AllCellTypes,
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllObjectParameters()));

TEST_P(SerializerServiceTests_AllCellTypes, objectWithEmptyGenome)
{
    auto objectParameter = GetParam();

    Desc data;
    if (objectParameter.objectType == ObjectType_Cell) {
        data.addCreature({_descriptionTestDataFactory->createNonDefaultObjectDesc(objectParameter)}, CreatureDesc(), GenomeDesc());
    } else {
        data.objects({_descriptionTestDataFactory->createNonDefaultObjectDesc(objectParameter)});
    }


    testSerializationAndDeserialization(data);
}

using NodeParameter = DescriptionTestDataFactory::NodeParameter;
class SerializerServiceTests_AllNodeTypes
    : public SerializerServiceTests
    , public testing::WithParamInterface<NodeParameter>
{};

INSTANTIATE_TEST_SUITE_P(
    SerializerServiceTests_AllNodeTypes,
    SerializerServiceTests_AllNodeTypes,
    ::testing::ValuesIn(DescriptionTestDataFactory::get().getAllNodeParameters()));

TEST_P(SerializerServiceTests_AllNodeTypes, objectWithNonEmptyGenome)
{
    auto nodeParameter = GetParam();

    auto [creature, genome] = _descriptionTestDataFactory->createNonDefaultCreatureDesc(nodeParameter);

    auto data = Desc().addCreature({ObjectDesc()}, creature, genome);

    testSerializationAndDeserialization(data);
}
