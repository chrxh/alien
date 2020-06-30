#include "IntegrationGpuTestFramework.h"
#include "ModelBasic/QuantityConverter.h"

class SensorGpuTests : public IntegrationGpuTestFramework
{
public:
    SensorGpuTests()
        : IntegrationGpuTestFramework({300, 300})
    {}

    virtual ~SensorGpuTests() = default;

protected:
    virtual void SetUp();

    struct TestParameters
    {
        MEMBER_DECLARATION(TestParameters, Enums::SensorIn::Type, command, Enums::SensorIn::DO_NOTHING);
        MEMBER_DECLARATION(TestParameters, boost::optional<float>, angle, boost::none);
        MEMBER_DECLARATION(TestParameters, int, minSize, 0);
        MEMBER_DECLARATION(TestParameters, int, maxSize, 0);
        MEMBER_DECLARATION(TestParameters, QVector2D, relPositionOfCluster, QVector2D{});
        MEMBER_DECLARATION(TestParameters, IntVector2D, sizeOfCluster, IntVector2D{});
        MEMBER_DECLARATION(TestParameters, IntVector2D, repetitionOfCluster, (IntVector2D{1, 1}));
    };
    struct Expectations
    {
        MEMBER_DECLARATION(Expectations, Enums::SensorOut::Type, command, Enums::SensorOut::NOTHING_FOUND);
        MEMBER_DECLARATION(Expectations, float, approxAngle, 0);
    };
    void runStandardTest(TestParameters const& testParameters, Expectations const& expectations) const;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void SensorGpuTests::SetUp()
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellFunctionSensorRange = 50;
    _context->setSimulationParameters(_parameters);
}

void SensorGpuTests::runStandardTest(TestParameters const& testParameters, Expectations const& expectations) const
{
    auto origCluster = createHorizontalCluster(2, QVector2D{}, QVector2D{}, 0);
    origCluster.angle = 30;
    auto& origFirstCell = origCluster.cells->at(0);
    origFirstCell.tokenBranchNumber = 0;
    auto& origSecondCell = origCluster.cells->at(1);
    origSecondCell.tokenBranchNumber = 1;
    origSecondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::SENSOR);
    auto origToken = createSimpleToken();
    auto& origTokenData = *origToken.data;
    origTokenData[Enums::Sensor::INPUT] = testParameters._command;
    if (testParameters._angle) {
        origTokenData[Enums::Sensor::INOUT_ANGLE] = QuantityConverter::convertAngleToData(*testParameters._angle);
    }
    origTokenData[Enums::Sensor::IN_MIN_MASS] = testParameters._minSize;
    origTokenData[Enums::Sensor::IN_MAX_MASS] = testParameters._maxSize;
    origFirstCell.addToken(origToken);

    DataDescription origData;
    origData.addCluster(origCluster);
    for (int x = 0; x < testParameters._repetitionOfCluster.x; ++x) {
        for (int y = 0; y < testParameters._repetitionOfCluster.y; ++y) {
            origData.addCluster(createRectangularCluster(
                testParameters._sizeOfCluster,
                testParameters._relPositionOfCluster
                    + QVector2D{toFloat(x * testParameters._sizeOfCluster.x), toFloat(y * testParameters._sizeOfCluster.y)},
                QVector2D{}));
        }
    }

    IntegrationTestHelper::updateData(_access, origData);
    IntegrationTestHelper::runSimulation(1, _controller);

    auto const data = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, data);

    auto const cellByCellId = IntegrationTestHelper::getCellByCellId(data);
    auto const& secondCell = cellByCellId.at(origSecondCell.id);

    EXPECT_EQ(1, secondCell.tokens->size());

    auto const& token = secondCell.tokens->at(0);
    EXPECT_EQ(expectations._command, token.data->at(Enums::Sensor::OUTPUT));

    auto const expectedMass = testParameters._sizeOfCluster.x * testParameters._sizeOfCluster.y;
    if (Enums::SensorOut::CLUSTER_FOUND == token.data->at(Enums::Sensor::OUTPUT)) {
        EXPECT_EQ(expectedMass, static_cast<unsigned char>(token.data->at(Enums::Sensor::OUT_MASS)));

        auto actualAngle = QuantityConverter::convertDataToAngle(token.data->at(Enums::Sensor::INOUT_ANGLE));
        auto angleDiff = expectations._approxAngle - actualAngle;
        if (angleDiff >= 360) {
            angleDiff -= 360;
        }
        if (angleDiff <= -360) {
            angleDiff += 360;
        }
        EXPECT_TRUE(abs(angleDiff) < 30);
    }
}

TEST_F(SensorGpuTests, testDoNothing)
{
    runStandardTest(
        TestParameters()
        .command(Enums::SensorIn::DO_NOTHING)
        .minSize(9)
        .relPositionOfCluster({ 20, 20 })
        .sizeOfCluster({ 3, 3 })
        .repetitionOfCluster({ 10, 10 }),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchVicinity_success)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_VICINITY)
            .minSize(9)
            .relPositionOfCluster({20, 20})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::CLUSTER_FOUND).approxAngle(-135));
}

TEST_F(SensorGpuTests, testSearchVicinity_scanMassTooSmall)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_VICINITY)
            .minSize(10)
            .relPositionOfCluster({20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchVicinity_scanMassTooLarge)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_VICINITY)
            .maxSize(8)
            .relPositionOfCluster({20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchVicinity_scanMassToFar)
{
    auto const sensorRange = _parameters.cellFunctionSensorRange;
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_VICINITY)
            .relPositionOfCluster({sensorRange + 20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchByAngle_success)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_BY_ANGLE)
            .angle(-135)
            .minSize(9)
            .relPositionOfCluster({20, 20})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::CLUSTER_FOUND).approxAngle(-135));
}

TEST_F(SensorGpuTests, testSearchByAngle_wrongAngle)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_BY_ANGLE)
            .angle(170)
            .minSize(9)
            .relPositionOfCluster({20, 20})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchByAngle_scanMassTooLarge)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_BY_ANGLE)
            .angle(-135)
            .maxSize(8)
            .relPositionOfCluster({20, 20})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchByAngle_scanMassToFar)
{
    auto const sensorRange = _parameters.cellFunctionSensorRange;
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_BY_ANGLE)
            .angle(-135)
            .relPositionOfCluster({sensorRange + 20, sensorRange + 20})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchFromCenter_success)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_FROM_CENTER)
            .angle(-135)  //should be ignored
            .minSize(9)
            .relPositionOfCluster({20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::CLUSTER_FOUND).approxAngle(180));
}

TEST_F(SensorGpuTests, testSearchFromCenter_scanMassTooLarge)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_FROM_CENTER)
            .angle(-135)    //should be ignored
            .maxSize(8)
            .relPositionOfCluster({20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchFromCenter_scanMassToFar)
{
    auto const sensorRange = _parameters.cellFunctionSensorRange;
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_FROM_CENTER)
            .angle(-135)    //should be ignored
            .relPositionOfCluster({sensorRange + 20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchTowardCenter_success)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_TOWARD_CENTER)
            .angle(-135)  //should be ignored
            .minSize(9)
            .relPositionOfCluster({-20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::CLUSTER_FOUND).approxAngle(0));
}

TEST_F(SensorGpuTests, testSearchTowardCenter_scanMassTooLarge)
{
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_TOWARD_CENTER)
            .angle(-135)    //should be ignored
            .maxSize(8)
            .relPositionOfCluster({-20, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}

TEST_F(SensorGpuTests, testSearchTowardCenter_scanMassToFar)
{
    auto const sensorRange = _parameters.cellFunctionSensorRange;
    runStandardTest(
        TestParameters()
            .command(Enums::SensorIn::SEARCH_TOWARD_CENTER)
            .angle(-135)    //should be ignored
            .relPositionOfCluster({-sensorRange - 40, 0})
            .sizeOfCluster({3, 3})
            .repetitionOfCluster({10, 10}),
        Expectations().command(Enums::SensorOut::NOTHING_FOUND));
}
