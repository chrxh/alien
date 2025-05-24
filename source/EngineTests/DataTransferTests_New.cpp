#include <gtest/gtest.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"
#include "IntegrationTestFramework.h"


class DataTransferTests_New : public IntegrationTestFramework
{
public:
    DataTransferTests_New()
        : IntegrationTestFramework()
    {}

};

TEST_F(DataTransferTests_New, singleCellWithoutGenome)
{
    CollectionDescription data;
    NeuralNetworkDescription nn;
    nn.weight(2, 1, 1.0f);
    data.addCell(CellDescription()
                     .neuralNetwork(nn)
                     .id(1)
                     .pos({2.0f, 4.0f})
                     .vel({0.5f, 1.0f})
                     .energy(100.0f)
                     .age(1)
                     .color(2)
                     .barrier(true)
                     .livingState(false)
                     .creatureId(3534)
                     .signal({1, 0, -1, 0, 0, 0, 0, 0}));

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}

class DataTransferTests_AllCellTypeGenome_New
    : public DataTransferTests_New
    , public testing::WithParamInterface<CellTypeGenome>
{
protected:
    CellTypeGenomeDescription_New createSomeCellTypeGenomeDescription(CellTypeGenome type)
    {
        switch (type) {
        case CellTypeGenome_Base:
            return BaseGenomeDescription();
        case CellTypeGenome_Depot:
            return DepotGenomeDescription();
        case CellTypeGenome_Constructor:
            return ConstructorGenomeDescription_New().autoTriggerInterval(7).constructionActivationTime(4).constructionAngle1(34.4f).constructionAngle2(-45.5f);
        case CellTypeGenome_Sensor:
            return SensorGenomeDescription().autoTriggerInterval(3).restrictToColor(5).minRange(34).maxRange(67).minDensity(0.25f).restrictToMutants(
                SensorRestrictToMutants_RestrictToLessComplexMutants);
        case CellTypeGenome_Oscillator:
            return OscillatorGenomeDescription().autoTriggerInterval(27).alternationInterval(45);
        case CellTypeGenome_Attacker:
            return AttackerGenomeDescription();
        case CellTypeGenome_Injector:
            return InjectorGenomeDescription_New();
        case CellTypeGenome_Muscle:
            return MuscleGenomeDescription().mode(ManualCrawlingGenomeDescription().frontBackVelRatio(0.4f).maxDistanceDeviation(0.3f));
        case CellTypeGenome_Defender:
            return DefenderGenomeDescription().mode(DefenderMode_DefendAgainstInjector);
        case CellTypeGenome_Reconnector:
            return ReconnectorGenomeDescription().restrictToColor(4).restrictToMutants(ReconnectorRestrictToMutants_RestrictToMoreComplexMutants);
        case CellTypeGenome_Detonator:
            return DetonatorGenomeDescription().countdown(23);
        default:
            return CellTypeGenomeDescription_New();
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    DataTransferTests_AllCellTypeGenome_New,
    DataTransferTests_AllCellTypeGenome_New,
    ::testing::Values(
        CellTypeGenome_Base,
        CellTypeGenome_Depot,
        CellTypeGenome_Constructor,
        CellTypeGenome_Sensor,
        CellTypeGenome_Oscillator,
        CellTypeGenome_Attacker,
        CellTypeGenome_Injector,
        CellTypeGenome_Muscle,
        CellTypeGenome_Defender,
        CellTypeGenome_Reconnector,
        CellTypeGenome_Detonator));

TEST_P(DataTransferTests_AllCellTypeGenome_New, singleCellWithGenome)
{
    auto cellType = GetParam();

    auto cellTypeGenomeDesc = createSomeCellTypeGenomeDescription(cellType);

    NeuralNetworkDescription nn1;
    nn1.weight(2, 1, 1.0f);
    NeuralNetworkGenomeDescription nn2;
    nn2.weight(1, 3, -1.0f);

    auto data = CollectionDescription().addCreature(
        GenomeDescription_New().id(1).genes({GeneDescription().nodes({NodeDescription().neuralNetwork(nn2).cellTypeData(cellTypeGenomeDesc)})}),
        {CellDescription()
             .neuralNetwork(nn1)
             .id(1)
             .pos({2.0f, 4.0f})
             .vel({0.5f, 1.0f})
             .energy(100.0f)
             .age(1)
             .color(2)
             .barrier(true)
             .livingState(false)
             .creatureId(3534)
             .signal({1, 0, -1, 0, 0, 0, 0, 0})});

    _simulationFacade->setSimulationData(data);
    auto actualData = _simulationFacade->getSimulationData();

    EXPECT_TRUE(compare(data, actualData));
}
