#include <gtest/gtest.h>

#include <Base/Math.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "IntegrationTestFramework.h"

class DigestorTests : public IntegrationTestFramework
{
public:
    DigestorTests()
        : IntegrationTestFramework()
    {
        _parameters.innerFriction.value = 0;
        _parameters.friction.baseValue = 0;
        for (int i = 0; i < MAX_COLORS; ++i) {
            _parameters.radiationType1_strength.baseValue[i] = 0;
        }
        _simulationFacade->setSimulationParameters(_parameters);
    }

    ~DigestorTests() = default;
};

TEST_F(DigestorTests, conversion_noEnergyConversion)
{
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(0)
            .pos({100.0f, 100.0f})
            .type(CellDesc().cellType(DigestorDesc().setRawEnergyConversionRate(0.0f)).rawEnergy(100.0f)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origDigestor = data.getObjectRef(0);
    auto actualDigestor = actualData.getObjectRef(0);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(approxCompare(actualDigestor.getCellRef()._rawEnergy, origDigestor.getCellRef()._rawEnergy));
    EXPECT_TRUE(approxCompare(actualDigestor.getCellRef()._usableEnergy, origDigestor.getCellRef()._usableEnergy));
}

TEST_F(DigestorTests, conversion_highEnergyConversionRate)
{
    auto data = Desc().addCreature({
        ObjectDesc()
            .id(0)
            .pos({100.0f, 100.0f})
            .type(CellDesc().cellType(DigestorDesc().setRawEnergyConversionRate(1.0f)).rawEnergy(100.0f)),
    });

    _simulationFacade->setSimulationData(data);
    _simulationFacade->calcTimesteps(1);

    auto actualData = _simulationFacade->getSimulationData();

    auto origDigestor = data.getObjectRef(0);
    auto actualDigestor = actualData.getObjectRef(0);

    EXPECT_TRUE(approxCompare(getEnergy(data), getEnergy(actualData)));
    EXPECT_TRUE(actualDigestor.getCellRef()._rawEnergy < origDigestor.getCellRef()._rawEnergy - NEAR_ZERO);
    EXPECT_TRUE(actualDigestor.getCellRef()._usableEnergy > origDigestor.getCellRef()._usableEnergy + NEAR_ZERO);
}
