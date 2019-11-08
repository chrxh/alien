#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/copy.hpp>

#include "Base/ServiceLocator.h"
#include "IntegrationGpuTestFramework.h"
#include "ModelBasic/CellComputerCompiler.h"
#include "ModelBasic/QuantityConverter.h"

class ConstructorGpuTests : public IntegrationGpuTestFramework
{
public:
    ConstructorGpuTests(
        IntVector2D const& universeSize = { 900, 600 }, 
        optional<ModelGpuData> const& modelData = boost::none)
        : IntegrationGpuTestFramework(universeSize, modelData)
    {}

    virtual ~ConstructorGpuTests() = default;

protected:
    virtual void SetUp();

    enum class WithSeparation
    {
        No,
        Yes
    };
    QVector2D constructorPositionForHorizontalClusterAfterCreation(
        vector<QVector2D> constructionSite,
        QVector2D constructor,
        vector<QVector2D> remainingCells,
        float distanceBetweenOffspringToConstructionSite,
        WithSeparation withSeparation) const;

    struct TokenForConstructionParameters
    {
        MEMBER_DECLARATION(TokenForConstructionParameters, optional<float>, energy, boost::none);
        MEMBER_DECLARATION(
            TokenForConstructionParameters,
            Enums::ConstrIn::Type,
            constructionInput,
            Enums::ConstrIn::DO_NOTHING);
        MEMBER_DECLARATION(
            TokenForConstructionParameters,
            Enums::ConstrInOption::Type,
            constructionOption,
            Enums::ConstrInOption::STANDARD);
        MEMBER_DECLARATION(
            TokenForConstructionParameters,
            Enums::CellFunction::Type,
            cellFunctionType,
            Enums::CellFunction::COMPUTER);
        MEMBER_DECLARATION(TokenForConstructionParameters, int, cellBranchNumber, 0);
        MEMBER_DECLARATION(TokenForConstructionParameters, int, maxConnections, 0);
        MEMBER_DECLARATION(TokenForConstructionParameters, QByteArray, staticData, QByteArray());
        MEMBER_DECLARATION(TokenForConstructionParameters, QByteArray, mutableData, QByteArray());
        MEMBER_DECLARATION(TokenForConstructionParameters, float, angle, 0.0f);
        MEMBER_DECLARATION(TokenForConstructionParameters, float, distance, 1.0f);
    };
    TokenDescription createTokenForConstruction(TokenForConstructionParameters tokenParameters) const;

    struct TestResult
    {
        QVector2D movementOfCenter;
        
        TokenDescription origToken;
        CellDescription origSourceCell;
        CellDescription origConstructorCell;
        vector<CellDescription> origConstructor;
        vector<CellDescription>
            origConstructionSite;  //ordered by assumed construction process where the front is the newest cell

        TokenDescription token;
        optional<CellDescription> sourceCell;  //possibly be destroyed
        CellDescription constructorCell;
        vector<CellDescription> constructionSite;

        optional<CellDescription> getFirstCellOfOrigConstructionSite() const;
        optional<CellDescription> getSecondCellOfOrigConstructionSite() const;

        optional<CellDescription> getFirstCellOfConstructionSiteAfterCreation() const;
        optional<CellDescription> getSecondCellOfConstructionSiteAfterCreation() const;
        optional<CellDescription> getThirdCellOfConstructionSiteAfterCreation() const;
        
        optional<CellDescription> getCellOfConstructionSite(uint64_t id) const;
    };
    struct StartConstructionOnHorizontalClusterTestParameters
    {
        MEMBER_DECLARATION(
            StartConstructionOnHorizontalClusterTestParameters,
            optional<float>,
            horizontalObstacleAt,
            boost::none);
        MEMBER_DECLARATION(
            StartConstructionOnHorizontalClusterTestParameters,
            TokenDescription,
            token,
            TokenDescription());
        MEMBER_DECLARATION(StartConstructionOnHorizontalClusterTestParameters, int, maxConnectionsOfConstructor, 2);
    };
    TestResult runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters const& parameters) const;

    struct StartConstructionOnWedgeClusterTestParameters
    {
        MEMBER_DECLARATION(StartConstructionOnWedgeClusterTestParameters, TokenDescription, token, TokenDescription());
        MEMBER_DECLARATION(StartConstructionOnWedgeClusterTestParameters, float, wedgeAngle, 90.0f);
        MEMBER_DECLARATION(StartConstructionOnWedgeClusterTestParameters, float, clusterAngle, 0.0f);
        MEMBER_DECLARATION(
            StartConstructionOnWedgeClusterTestParameters,
            optional<QVector2D>,
            referencePosition,
            boost::none);
    };
    TestResult runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters const& parameters) const;

    struct StartConstructionOnTriangleClusterTestParameters
    {
        MEMBER_DECLARATION(StartConstructionOnTriangleClusterTestParameters, TokenDescription, token, TokenDescription());
        MEMBER_DECLARATION(StartConstructionOnTriangleClusterTestParameters, int, tokensOnSource1, 1);
        MEMBER_DECLARATION(StartConstructionOnTriangleClusterTestParameters, int, tokensOnSource2, 0);
        MEMBER_DECLARATION(StartConstructionOnTriangleClusterTestParameters, int, tokensOnSource3, 0);
    };
    TestResult runStartConstructionOnTriangleClusterTest(
        StartConstructionOnTriangleClusterTestParameters const& parameters) const;

    struct SecondCellConstructionOnLineClusterTestParameters
    {
        MEMBER_DECLARATION(
            SecondCellConstructionOnLineClusterTestParameters,
            TokenDescription,
            tokenOnSourceCell,
            TokenDescription());
        MEMBER_DECLARATION(
            SecondCellConstructionOnLineClusterTestParameters,
            optional<TokenDescription>,
            tokenOnConstructionSite,
            boost::none);
        MEMBER_DECLARATION(
            SecondCellConstructionOnLineClusterTestParameters,
            optional<float>,
            horizontalObstacleAt,
            boost::none);
        MEMBER_DECLARATION(
            SecondCellConstructionOnLineClusterTestParameters,
            float,
            angleOfConstructionSite,
            180.0f);
    };
    TestResult runSecondConstructionOnLineClusterTest(SecondCellConstructionOnLineClusterTestParameters const& parameters) const;
    
    TestResult runSecondCellConstructionOnSelfTouchingClusterTest(TokenDescription const& token, int cellLength) const;

    struct FurtherCellConstructionOnLineClusterTestParameters {
        MEMBER_DECLARATION(
            FurtherCellConstructionOnLineClusterTestParameters,
            TokenDescription,
            tokenOnSourceCell,
            TokenDescription());
        MEMBER_DECLARATION(
            FurtherCellConstructionOnLineClusterTestParameters,
            optional<float>,
            verticalObstacleAt,
            boost::none);
        MEMBER_DECLARATION(
            FurtherCellConstructionOnLineClusterTestParameters,
            int,
            additionalCellsOnConstructor,
            0);

        struct CellProperties
        {
            float angle;
            int maxConnection;  //0 = automatic
            optional<TokenDescription> token;
            optional<int> tokenBranchNumber;
        };
        FurtherCellConstructionOnLineClusterTestParameters& propertiesOfConstructionSite(vector<CellProperties> const& value)
        {
            _propertiesOfConstructionSite = value;
            return *this;
        }
        vector<CellProperties> _propertiesOfConstructionSite{{180.0f, 0}, {180.0f, 0}};
    };
    TestResult runFurtherCellConstructionOnLineClusterTest(FurtherCellConstructionOnLineClusterTestParameters const& parameters) const;

    TestResult runConstructionSiteConnectedToConstructorTwiceTest(TokenDescription const& token) const;

    struct MassiveParallelClustersTestParameters
    {
        MEMBER_DECLARATION(MassiveParallelClustersTestParameters, int, clusterLen, 100);
        MEMBER_DECLARATION(MassiveParallelClustersTestParameters, int, numClusters, 10);
        MEMBER_DECLARATION(MassiveParallelClustersTestParameters, int, distanceBetweenClusters, 6);
    };
    struct MassiveParallelClustersTestResult {
        vector<int> numCellsPerCluster;
    };
    MassiveParallelClustersTestResult runMassiveParallelClustersTest(
        MassiveParallelClustersTestParameters const& parameters) const;

    struct Expectations
    {
        MEMBER_DECLARATION(Expectations, Enums::ConstrOut::Type, tokenOutput, Enums::ConstrOut::SUCCESS);
        MEMBER_DECLARATION(Expectations, optional<QVector2D>, relPosOfFirstCellOfConstructionSite, boost::none);
        MEMBER_DECLARATION(Expectations, optional<TokenDescription>, constructedToken, boost::none);
        MEMBER_DECLARATION(Expectations, bool, destruction, false);
    };

    class _ResultChecker
    {
    public:
        _ResultChecker(SimulationParameters const& parameters, SpaceProperties* spaceProp)
            : _parameters(parameters), _spaceProp(spaceProp)
        {}

        void check(TestResult const& testResult, Expectations const& expectations) const;

    private:
        void checkIfDestruction(TestResult const& testResult, Expectations const& expectations) const;
        void checkIfNoDestruction(TestResult const& testResult, Expectations const& expectations) const;
        void checkIfNoDestructionAndSuccess(TestResult const& testResult, Expectations const& expectations) const;
        void checkIfNoDestructionAndSuccessRotationOnly(TestResult const& testResult, Expectations const& expectations) const;

        void checkCellPositionAfterCreation(TestResult const& testResult, Expectations const& expectations) const;
        void checkCellAttributesAfterCreation(TestResult const& testResult) const;
        void checkCellConnectionsAfterCreation(TestResult const& testResult) const;
        void checkConstructedTokenAfterCreation(TestResult const& testResult, Expectations const& expectations) const;

        void checkCellPositionAfterRotation(TestResult const& testResult, Expectations const& expectations) const;
        void checkCellConnectionsAfterRotation(TestResult const& testResult, Expectations const& expectations) const;

        void checkTokenMovement(TestResult const& testResult) const;
        void checkConstructionSiteDistances(TestResult const& testResult) const;
        struct AngularMasses {
            float constructionSite;
            float constructor;
        };
        AngularMasses calcAngularMasses(TestResult const& testResult) const;

        Enums::ConstrIn::Type getConstrIn(TokenDescription const& token) const;
        Enums::ConstrOut::Type getConstrOut(TokenDescription const& token) const;
    private:
        SimulationParameters _parameters;
        SpaceProperties* _spaceProp;
    };
    using ResultChecker = boost::shared_ptr<_ResultChecker>;

    static bool isFinished(TokenDescription const& token);
    static bool isReduced(TokenDescription const& token);
    static bool isSeparated(TokenDescription const& token);
    static bool isAutomaticMaxConnections(TokenDescription const& token);

protected:
    ResultChecker _resultChecker;
    float _offspringDistance;
};

namespace
{
    ModelGpuData getModelGpuDataWithHighBlockCount()
    {
        ModelGpuData result;
        result.setNumThreadsPerBlock(16);
        result.setNumBlocks(64*8);
        result.setNumClusterPointerArrays(1);
        result.setMaxClusters(100000);
        result.setMaxCells(500000);
        result.setMaxParticles(500000);
        result.setMaxTokens(50000);
        result.setMaxCellPointers(500000 * 10);
        result.setMaxClusterPointers(100000 * 10);
        result.setMaxParticlePointers(500000 * 10);
        result.setMaxTokenPointers(50000 * 10);
        result.setDynamicMemorySize(100000000);
        return result;
    }
}

class ConstructorGpuWithHighBlockCountTests : public ConstructorGpuTests
{
public:
    ConstructorGpuWithHighBlockCountTests()
        : ConstructorGpuTests({ 64 * 8 * 6 * 4 + 100, 20 }, getModelGpuDataWithHighBlockCount())
    { }

    virtual ~ConstructorGpuWithHighBlockCountTests() = default;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

void ConstructorGpuTests::SetUp()
{
    _parameters.radiationProb = 0;  //exclude radiation
    _parameters.cellFunctionConstructorOffspringCellDistance = 1;
    _context->setSimulationParameters(_parameters);

    _resultChecker = boost::make_shared<_ResultChecker>(_parameters, _spaceProp);
    _offspringDistance = _parameters.cellFunctionConstructorOffspringCellDistance;
}

auto ConstructorGpuTests::runStartConstructionOnHorizontalClusterTest(
    StartConstructionOnHorizontalClusterTestParameters const& parameters) const -> TestResult
{
    DataDescription origData;
    auto cluster = createHorizontalCluster(2, QVector2D{10.5, 10.5}, QVector2D{}, 0);

    auto& firstCell = cluster.cells->at(0);
    firstCell.tokenBranchNumber = 0;
    firstCell.maxConnections = 1;
    firstCell.addToken(parameters._token);

    auto& secondCell = cluster.cells->at(1);
    secondCell.tokenBranchNumber = 1;
    secondCell.maxConnections = parameters._maxConnectionsOfConstructor;
    secondCell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);

    origData.addCluster(cluster);

    set<std::pair<float, float>> obstaclePositions;
    if (parameters._horizontalObstacleAt) {

        //following calculation only works for 0-angle
        CHECK(0 == parameters._token.data->at(Enums::Constr::INOUT_ANGLE));

        auto const withSeparation = isSeparated(parameters._token);

        auto const distance =
            QuantityConverter::convertDataToDistance(parameters._token.data->at(Enums::Constr::IN_DIST));
        auto const estimatedConstructorAbsPos = constructorPositionForHorizontalClusterAfterCreation(
            {}, {11, 10.5}, {{10, 10.5}}, distance, withSeparation ? WithSeparation::Yes : WithSeparation::No);
        auto const obstacleCellAbsPos = estimatedConstructorAbsPos + QVector2D{*parameters._horizontalObstacleAt, 0};

        QVector2D obstacleCenterPos;
        if (*parameters._horizontalObstacleAt > 0) {
            obstacleCenterPos = obstacleCellAbsPos + QVector2D{1.5f + _parameters.cellMinDistance / 2, 0};
        } else {
            obstacleCenterPos = obstacleCellAbsPos - QVector2D{1.5f + _parameters.cellMinDistance / 2, 0};
        }
        auto obstacle = createHorizontalCluster(4, obstacleCenterPos, QVector2D{}, 0);
        origData.addCluster(obstacle);
        for (auto const& cell : *obstacle.cells) {
            obstaclePositions.emplace(cell.pos->x(), cell.pos->y());
        }
    }

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});

    check(origData, newData);

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    std::unordered_map<uint64_t, CellDescription> newCellsWithoutObstacleByCellId;
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()) {
            newCellsWithoutObstacleByCellId.insert_or_assign(newCell.id, newCell);
        }
    }

    QVector2D newCenter;
    for (auto const& newCell : newCellsWithoutObstacleByCellId | boost::adaptors::map_values) {
        newCenter += *newCell.pos;
    }
    newCenter /= newCellsWithoutObstacleByCellId.size();

    TestResult result;
    result.movementOfCenter = newCenter - *cluster.pos;

    auto const& newSecondCell = newCellByCellId.at(secondCell.id);
    auto const& newToken = newSecondCell.tokens->at(0);

    result.origToken = parameters._token;
    result.token = newToken;
    result.origSourceCell = firstCell;
    if (newCellByCellId.find(firstCell.id) != newCellByCellId.end()) {
        result.sourceCell = newCellByCellId.at(firstCell.id);
    }
    result.origConstructorCell = secondCell;
    result.constructorCell = newSecondCell;
    result.origConstructor = *cluster.cells;

    std::list<CellDescription> remainingCells;
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (newCell.id != firstCell.id && newCell.id != secondCell.id
            && obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()) {
            remainingCells.push_back(newCell);
        }
    }
    for (auto const& remainingCell : remainingCells) {
        if (remainingCell.pos->x() >= result.constructorCell.pos->x()
                - 0.1) {  //rough estimation of the position of constructed cell (right hand side of constructor)
            result.constructionSite.emplace_back(remainingCell);
        }
    }

    EXPECT_GE(1, result.constructionSite.size());

    return result;
}

auto ConstructorGpuTests::runStartConstructionOnWedgeClusterTest(
    StartConstructionOnWedgeClusterTestParameters const& parameters) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const refPos = parameters._referencePosition.get_value_or(QVector2D{10.5f, 10.5f});
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = Physics::unitVectorOfAngle(parameters._clusterAngle + 270 + parameters._wedgeAngle / 2);
    auto const relPos2 = QVector2D{0, 0};
    auto const relPos3 = Physics::unitVectorOfAngle(parameters._clusterAngle + 270 - parameters._wedgeAngle / 2);
    auto const cellId1 = _numberGen->getId();
    auto const cellId2 = _numberGen->getId();
    auto const cellId3 = _numberGen->getId();
    cluster.addCells({CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos1)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId2})
                          .setTokenBranchNumber(0)
                          .setId(cellId1)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(parameters._token),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos2)
                          .setMaxConnections(2)
                          .setConnectingCells({cellId1, cellId3})
                          .setTokenBranchNumber(1)
                          .setId(cellId2)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR)),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos3)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId2})
                          .setTokenBranchNumber(2)
                          .setId(cellId3)
                          .setCellFeature(CellFeatureDescription())});
    auto const& cell1 = cluster.cells->at(0);
    auto const& cell2 = cluster.cells->at(1);

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, newData);

    auto const& newCluster = newData.clusters->at(0);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    auto const& newCell2 = newCellByCellId.at(cellId2);
    auto const& newToken = newCell2.tokens->at(0);

    result.origToken = parameters._token;
    result.token = newToken;
    result.origSourceCell = cell1;
    if (newCellByCellId.find(cellId1) != newCellByCellId.end()) {
        result.sourceCell = newCellByCellId.at(cellId1);
    }
    result.origConstructorCell = cell2;
    result.constructorCell = newCell2;
    result.origConstructor = *cluster.cells;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    if (!newCellByCellId.empty()) {
        result.constructionSite.emplace_back(newCellByCellId.begin()->second);
    }

    return result;
}

auto ConstructorGpuTests::runStartConstructionOnTriangleClusterTest(
    StartConstructionOnTriangleClusterTestParameters const& parameters) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const refPos = QVector2D{10, 10};
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = QVector2D{0, -1};
    auto const relPos2 = QVector2D{-1, 0};
    auto const relPos3 = QVector2D{0, 1};
    auto const relPos4 = QVector2D{0, 0};
    auto const cellId1 = _numberGen->getId();
    auto const cellId2 = _numberGen->getId();
    auto const cellId3 = _numberGen->getId();
    auto const cellId4 = _numberGen->getId();
    cluster.addCells({CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos1)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId1)
                          .setCellFeature(CellFeatureDescription()),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos2)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId2)
                          .setCellFeature(CellFeatureDescription()),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos3)
                          .setMaxConnections(1)
                          .setConnectingCells({cellId4})
                          .setTokenBranchNumber(0)
                          .setId(cellId3)
                          .setCellFeature(CellFeatureDescription()),
                      CellDescription()
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos4)
                          .setMaxConnections(3)
                          .setConnectingCells({cellId1, cellId2, cellId3})
                          .setTokenBranchNumber(1)
                          .setId(cellId4)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR))});

    auto& cell1 = cluster.cells->at(0);
    auto& cell2 = cluster.cells->at(1);
    auto& cell3 = cluster.cells->at(2);
    auto& cell4 = cluster.cells->at(3);
    for (int i = 0; i < parameters._tokensOnSource1; ++i) {
        cell1.addToken(parameters._token);
    }
    for (int i = 0; i < parameters._tokensOnSource2; ++i) {
        cell2.addToken(parameters._token);
    }
    for (int i = 0; i < parameters._tokensOnSource3; ++i) {
        cell3.addToken(parameters._token);
    }

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, newData);

    EXPECT_EQ(1, newData.clusters->size());
    auto const& newCluster = newData.clusters->at(0);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    auto const& newCell4 = newCellByCellId.at(cellId4);
    auto const& newToken = newCell4.tokens->at(0);

    result.origToken = parameters._token;
    result.token = newToken;
    result.origSourceCell = cell1;
    result.sourceCell = newCellByCellId.at(cellId1);
    result.origConstructorCell = cell4;
    result.constructorCell = newCell4;
    result.origConstructor = *cluster.cells;

    newCellByCellId.erase(cellId1);
    newCellByCellId.erase(cellId2);
    newCellByCellId.erase(cellId3);
    newCellByCellId.erase(cellId4);
    for(auto const& cell : newCellByCellId | boost::adaptors::map_values) {
        result.constructionSite.emplace_back(cell);
    }

    return result;
}

auto ConstructorGpuTests::runSecondConstructionOnLineClusterTest(
    SecondCellConstructionOnLineClusterTestParameters const& parameters) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const refPos = QVector2D{ 10.5f, 10.5f };
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const relPos1 = QVector2D{ -1, 0 };
    auto const relPos2 = QVector2D{ 0, 0 };
    auto const relPos3 = Physics::unitVectorOfAngle(parameters._angleOfConstructionSite - 90.0f);
    auto const cellId1 = _numberGen->getId();
    auto const cellId2 = _numberGen->getId();
    auto const cellId3 = _numberGen->getId();
    cluster.addCells({CellDescription()
                          .setId(cellId1)
                          .setConnectingCells({cellId2})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos1)
                          .setMaxConnections(1)
                          .setTokenBranchNumber(0)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(parameters._tokenOnSourceCell),
                      CellDescription()
                          .setId(cellId2)
                          .setConnectingCells({cellId1, cellId3})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos2)
                          .setMaxConnections(2)
                          .setTokenBranchNumber(1)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR)),
                      CellDescription()
                          .setId(cellId3)
                          .setConnectingCells({cellId2})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + relPos3)
                          .setMaxConnections(1)
                          .setTokenBranchNumber(1)
                          .setFlagTokenBlocked(true)
                          .setCellFeature(CellFeatureDescription())});
    if (parameters._tokenOnConstructionSite) {
        cluster.cells->at(2).addToken(*parameters._tokenOnConstructionSite);
    }
    auto const& cell1 = cluster.cells->at(0);
    auto const& cell2 = cluster.cells->at(1);
    auto const& cell3 = cluster.cells->at(2);
    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    set<std::pair<float, float>> obstaclePositions;
    if (parameters._horizontalObstacleAt) {

        //following calculation only works for 0-angle
        CHECK(0 == parameters._tokenOnSourceCell.data->at(Enums::Constr::INOUT_ANGLE));

        auto const withSeparation = isSeparated(parameters._tokenOnSourceCell);
        auto const distance =
            QuantityConverter::convertDataToDistance(parameters._tokenOnSourceCell.data->at(Enums::Constr::IN_DIST));
        auto const estimatedConstructorAbsPos = constructorPositionForHorizontalClusterAfterCreation(
            {{11.5, 10.5}},
            {10.5, 10.5},
            {{9.5, 10.5}},
            distance,
            withSeparation ? WithSeparation::Yes : WithSeparation::No);
        auto const obstacleCellAbsPos = estimatedConstructorAbsPos + QVector2D{ *parameters._horizontalObstacleAt, 0 };

        QVector2D obstacleCenterPos;
        if (*parameters._horizontalObstacleAt > 0) {
            obstacleCenterPos = obstacleCellAbsPos + QVector2D{ 1.5f + _parameters.cellMinDistance / 2, 0 };
        }
        else {
            obstacleCenterPos = obstacleCellAbsPos - QVector2D{ 1.5f + _parameters.cellMinDistance / 2, 0 };
        }
        auto obstacle = createHorizontalCluster(4, obstacleCenterPos, QVector2D{}, 0);
        origData.addCluster(obstacle);
        for (auto const& cell : *obstacle.cells) {
            obstaclePositions.emplace(cell.pos->x(), cell.pos->y());
        }
    }

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, newData);

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    std::unordered_map<uint64_t, CellDescription> newCellsWithoutObstacleByCellId;
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()) {
            newCellsWithoutObstacleByCellId.insert_or_assign(newCell.id, newCell);
        }
    }

    QVector2D newCenter;
    for (auto const& newCell : newCellsWithoutObstacleByCellId | boost::adaptors::map_values) {
        newCenter += *newCell.pos;
    }
    newCenter /= newCellsWithoutObstacleByCellId.size();

    TestResult result;
    result.movementOfCenter = newCenter - *cluster.pos;

    auto const& newCell2 = newCellByCellId.at(cell2.id);
    auto const& newToken = newCell2.tokens->at(0);

    result.origToken = parameters._tokenOnSourceCell;
    result.token = newToken;
    result.origSourceCell = cell1;
    if (newCellByCellId.find(cell1.id) != newCellByCellId.end()) {
        result.sourceCell = newCellByCellId.at(cell1.id);
    }
    result.origConstructorCell = cell2;
    result.origConstructionSite.emplace_back(cell3);
    result.constructorCell = newCell2;
    result.origConstructor.emplace_back(cell1);
    result.origConstructor.emplace_back(cell2);

    std::vector<CellDescription> remainingCells;
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (newCell.id != cell1.id && newCell.id != cell2.id
            && obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()
            && newCell.pos->x() >= result.constructorCell.pos->x() - 0.1) {
            remainingCells.push_back(newCell);
        }
    }
    EXPECT_GE(2, remainingCells.size());

    result.constructionSite = std::move(remainingCells);

    return result;
}

auto ConstructorGpuTests::runSecondCellConstructionOnSelfTouchingClusterTest(
    TokenDescription const& token,
    int cellLength) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const refPos = QVector2D{ 10.5f, 10.5f };
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;

    vector<uint64_t> cellIds;
    for (int i = 0; i < cellLength + 4; ++i) {
        cellIds.emplace_back(_numberGen->getId());
    }
    cluster.addCells({CellDescription()  //construction site
                          .setId(cellIds[0])
                          .setConnectingCells({cellIds[1]})
                          .setEnergy(cellEnergy)
                          .setPos(refPos)
                          .setMaxConnections(1)
                          .setTokenBranchNumber(0)
                          .setCellFeature(CellFeatureDescription())
                          .setFlagTokenBlocked(true),
                      CellDescription()  //constructor
                          .setId(cellIds[1])
                          .setConnectingCells({cellIds[0], cellIds[2]})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + QVector2D{-1, 0})
                          .setMaxConnections(2)
                          .setTokenBranchNumber(1)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR)),
                      CellDescription()
                          .setId(cellIds[2])
                          .setConnectingCells({cellIds[1], cellIds[3]})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + QVector2D{-2, 0})
                          .setMaxConnections(2)
                          .setTokenBranchNumber(0)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(token)});
    for (int i = 0; i < cellLength; ++i) {
        cluster.addCell(CellDescription()
                            .setId(cellIds[3 + i])
                            .setConnectingCells({cellIds[2 + i], cellIds[4 + i]})
                            .setEnergy(cellEnergy)
                            .setPos(refPos + QVector2D{-2.0f + i, 1.0f })
                            .setMaxConnections(2)
                            .setTokenBranchNumber(0)
                            .setCellFeature(CellFeatureDescription()));
    }
    cluster.addCell(CellDescription()
                        .setId(cellIds[3 + cellLength])
                        .setConnectingCells({cellIds[2 + cellLength]})
                        .setEnergy(cellEnergy)
                        .setPos(refPos + QVector2D{-2.0f + cellLength - 1.0f, 0})
                        .setMaxConnections(1)
                        .setTokenBranchNumber(0)
                        .setCellFeature(CellFeatureDescription()));

    auto const& origCells = *cluster.cells;

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);

    auto const& newCluster = newData.clusters->at(0);

    TestResult result;
    result.movementOfCenter = *newCluster.pos - *cluster.pos;

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    auto const& newConstructor = newCellByCellId.at(cellIds[1]);
    auto const& newToken = newConstructor.tokens->at(0);

    result.origToken = token;
    result.token = newToken;
    result.origSourceCell = origCells[0];
    if (newCellByCellId.find(cellIds[0]) != newCellByCellId.end()) {
        result.sourceCell = newCellByCellId.at(cellIds[0]);
    }
    result.origConstructorCell = origCells[1];
    result.constructorCell = newConstructor;
    for (int i = 1; i <= 3 + cellLength; ++i) {
        result.origConstructor.emplace_back(origCells[i]);
        newCellByCellId.erase(cellIds[i]);
    }
    result.origConstructionSite.emplace_back(origCells[0]);
    for (auto const& cell : newCellByCellId | boost::adaptors::map_values) {
        result.constructionSite.emplace_back(cell);
    }

    return result;
}

auto ConstructorGpuTests::runFurtherCellConstructionOnLineClusterTest(
    FurtherCellConstructionOnLineClusterTestParameters const& parameters) const -> TestResult
{
    ClusterDescription cluster;
    cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

    auto const numCellsOfConstructionSite = parameters._propertiesOfConstructionSite.size();
    CHECK(1 <= numCellsOfConstructionSite);

    vector<uint64_t> cellIds;
    for (int i = 0; i < numCellsOfConstructionSite + 2 + parameters._additionalCellsOnConstructor; ++i) {
        cellIds.emplace_back(_numberGen->getId());
    }

    auto const refPos = QVector2D{ 10.5f, 10.5f };
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    for (int i = 0; i < parameters._additionalCellsOnConstructor; ++i) {
        list<uint64_t> connectingCells;
        connectingCells.push_back(cellIds[i + 1]);
        if (i > 0) {
            connectingCells.push_back(cellIds[i - 1]);
        }
        cluster.addCell(CellDescription().setId(cellIds[i])
            .setConnectingCells(connectingCells)
            .setEnergy(cellEnergy)
            .setPos(refPos + QVector2D{ static_cast<float>(i), 0 })
            .setMaxConnections(2)
            .setTokenBranchNumber(0)
            .setCellFeature(CellFeatureDescription()));
    }
    int offset = parameters._additionalCellsOnConstructor;

    list<uint64_t> connectingCellsOfNextCell;
    connectingCellsOfNextCell.push_back(cellIds[offset + 1]);
    if (parameters._additionalCellsOnConstructor > 0) {
        connectingCellsOfNextCell.push_back(cellIds[offset - 1]);
    }
    cluster.addCells({CellDescription()
                          .setId(cellIds[offset])
                          .setConnectingCells(connectingCellsOfNextCell)
                          .setEnergy(cellEnergy)
                          .setPos(refPos + QVector2D{ static_cast<float>(offset), 0 })
                          .setMaxConnections(connectingCellsOfNextCell.size())
                          .setTokenBranchNumber(0)
                          .setCellFeature(CellFeatureDescription())
                          .addToken(parameters._tokenOnSourceCell),
                      CellDescription()
                          .setId(cellIds[offset + 1])
                          .setConnectingCells({cellIds[offset], cellIds[offset + 2]})
                          .setEnergy(cellEnergy)
                          .setPos(refPos + QVector2D{ static_cast<float>(offset) + 1, 0})
                          .setMaxConnections(2)
                          .setTokenBranchNumber(1)
                          .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR))});

    auto lastAngle = 90.0f;     //abolute value with respect to {0, -1}
    auto lastPosition = refPos + QVector2D{ static_cast<float>(offset) + 1, 0 };
    for (int i = 0; i < numCellsOfConstructionSite; ++i) {
        list<uint64_t> connectingCells{ cellIds[offset + 1 + i] };
        if (i < numCellsOfConstructionSite - 1) {
            connectingCells.emplace_back(cellIds[offset + 3 + i]);
        }
        auto const tokenBlocked = 0 == i;
        auto cellProperties = parameters._propertiesOfConstructionSite[i];
        auto const newPosition =
            lastPosition + Physics::unitVectorOfAngle(lastAngle + cellProperties.angle - 180.0f);
        auto maxConnection = i < numCellsOfConstructionSite - 1 ? 2 : 1;
        if (0 < cellProperties.maxConnection) {
            maxConnection = cellProperties.maxConnection;
        }
        auto const tokenBranchNumber = cellProperties.tokenBranchNumber ? *cellProperties.tokenBranchNumber : 1;
        auto cell = CellDescription()
                        .setId(cellIds[offset + 2 + i])
                        .setConnectingCells(connectingCells)
                        .setFlagTokenBlocked(tokenBlocked)
                        .setEnergy(cellEnergy)
                        .setPos(newPosition)
                        .setMaxConnections(maxConnection)
                        .setTokenBranchNumber(tokenBranchNumber);
        if (cellProperties.token) {
            cell.addToken(*cellProperties.token);
        }
        cluster.addCell(cell);
        lastAngle = lastAngle + cellProperties.angle - 180.0f;
        lastPosition = newPosition;
    }
    auto const& origCells = *cluster.cells;

    cluster.setPos(cluster.getClusterPosFromCells());

    DataDescription origData;
    origData.addCluster(cluster);

    set<std::pair<float, float>> obstaclePositions;
    if (parameters._verticalObstacleAt) {
        auto const obstacle = createRectangularCluster(
            {10, 2}, *cluster.pos + QVector2D{0, -0.5f + *parameters._verticalObstacleAt}, QVector2D{});
        origData.addCluster(obstacle);
        for (auto const& cell : *obstacle.cells) {
            obstaclePositions.emplace(cell.pos->x(), cell.pos->y());
        }
    }

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, {{0, 0}, {_universeSize.x, _universeSize.y}});
    check(origData, newData);

    auto newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);

    std::unordered_map<uint64_t, CellDescription> newCellsWithoutObstacleByCellId;
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()) {
            newCellsWithoutObstacleByCellId.insert_or_assign(newCell.id, newCell);
        }
    }

    QVector2D newCenter;
    for (auto const& newCell : newCellsWithoutObstacleByCellId | boost::adaptors::map_values) {
        newCenter += *newCell.pos;
    }
    newCenter /= newCellsWithoutObstacleByCellId.size();

    TestResult result;
    result.movementOfCenter = newCenter - *cluster.pos;

    auto const& newConstructor = newCellByCellId.at(cellIds[offset + 1]);
    auto const& newToken = newConstructor.tokens->at(0);

    result.origToken = parameters._tokenOnSourceCell;
    result.token = newToken;
    result.origSourceCell = origCells[offset];
    if (newCellByCellId.find(cellIds[offset]) != newCellByCellId.end()) {
        result.sourceCell = newCellByCellId.at(cellIds[offset]);
    }
    result.origConstructorCell = origCells[offset + 1];
    result.constructorCell = newConstructor;
    for (int i = 0; i < offset + 2; ++i) {
        result.origConstructor.emplace_back(origCells[i]);
        newCellByCellId.erase(cellIds[i]);
    }
    for (int i = 0; i < numCellsOfConstructionSite; ++i) {
        result.origConstructionSite.emplace_back(origCells[offset + 2 + i]);
    }
    for (auto const& newCell : newCellByCellId | boost::adaptors::map_values) {
        if (obstaclePositions.find({ newCell.pos->x(), newCell.pos->y() }) == obstaclePositions.end()) {
            result.constructionSite.emplace_back(newCell);
        }
    }

    return result;
}

auto ConstructorGpuTests::runConstructionSiteConnectedToConstructorTwiceTest(TokenDescription const& token) const
    -> TestResult
{
    auto cluster = createRectangularCluster({ 2,2 }, QVector2D{ 10, 10 }, QVector2D());
    auto& firstCellOfConstructionSite = cluster.cells->at(1);
    auto& sourceCell = cluster.cells->at(2);
    auto& constructorCell = cluster.cells->at(3);

    sourceCell.addToken(token);
    sourceCell.setTokenBranchNumber(0);
    constructorCell.setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR));
    constructorCell.setTokenBranchNumber(1);
    firstCellOfConstructionSite.setFlagTokenBlocked(true);

    DataDescription origData;
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);

    auto const newCellByCellId = IntegrationTestHelper::getCellByCellId(newData);
    auto const newConstructorCell = newCellByCellId.at(constructorCell.id);

    TestResult result;
    result.token = newConstructorCell.tokens->at(0);
    result.movementOfCenter = *newData.clusters->at(0).pos - *cluster.pos;

    return result;
}

auto ConstructorGpuTests::runMassiveParallelClustersTest(MassiveParallelClustersTestParameters const& parameters) const
    -> MassiveParallelClustersTestResult
{
    auto const maxToken = _parameters.cellMaxToken;
    auto const cellEnergy = _parameters.cellFunctionConstructorOffspringCellEnergy;
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));

    auto const refPos = QVector2D{ 10, 10 };

    DataDescription origData;
    for (int clusterIndex = 0; clusterIndex < parameters._numClusters; ++clusterIndex) {
        ClusterDescription cluster;
        cluster.setId(_numberGen->getId()).setVel(QVector2D{}).setAngle(0).setAngularVel(0);

        std::vector<uint64_t> cellIds;
        for (int i = 0; i < parameters._clusterLen * 3; ++i) {
            cellIds.emplace_back(_numberGen->getId());
        }
        for (int i = 0; i < parameters._clusterLen; ++i) {
            std::list<uint64_t> connectingCells1{ cellIds[i * 3 + 1] };
            std::list<uint64_t> connectingCells2{ cellIds[i * 3], cellIds[i * 3] + 2 };
            std::list<uint64_t> connectingCells3{ cellIds[i * 3] + 1 };
            if (i > 0) {
                connectingCells1.emplace_back(cellIds[i * 3 - 3]);
            }
            if (i < parameters._clusterLen - 1) {
                connectingCells1.emplace_back(cellIds[i * 3 + 3]);
            }
            auto shiftX = static_cast<float>(clusterIndex * parameters._distanceBetweenClusters);
            cluster.addCells({CellDescription()
                                  .setId(cellIds[i * 3])
                                  .setConnectingCells(connectingCells1)
                                  .setMaxConnections(connectingCells1.size())
                                  .setEnergy(cellEnergy)
                                  .setPos(refPos + QVector2D{shiftX, static_cast<float>(i)})
                                  .setTokenBranchNumber(0)
                                  .setFlagTokenBlocked(false)
//                                  .addToken(token)
                                  .addToken(token),
                              CellDescription()
                                  .setId(cellIds[i * 3 + 1])
                                  .setConnectingCells(connectingCells2)
                                  .setMaxConnections(connectingCells2.size())
                                  .setEnergy(cellEnergy)
                                  .setPos(refPos + QVector2D{1 + shiftX, static_cast<float>(i)})
                                  .setTokenBranchNumber(1)
                                  .setFlagTokenBlocked(false)
                                  .setCellFeature(CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR)),
                              CellDescription()
                                  .setId(cellIds[i * 3 + 2])
                                  .setConnectingCells(connectingCells3)
                                  .setMaxConnections(connectingCells3.size())
                                  .setEnergy(cellEnergy)
                                  .setPos(refPos + QVector2D{2 + shiftX, static_cast<float>(i)})
                                  .setTokenBranchNumber(0)
                                  .setFlagTokenBlocked(true)});
        }
        cluster.setPos(cluster.getClusterPosFromCells());

        origData.addCluster(cluster);
    }

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);

    MassiveParallelClustersTestResult result;
    for (auto const& newCluster : *newData.clusters) {
        result.numCellsPerCluster.push_back(newCluster.cells->size());
    }
    return result;
}

TokenDescription ConstructorGpuTests::createTokenForConstruction(TokenForConstructionParameters tokenParameters) const
{
    auto token = createSimpleToken();
    auto& tokenData = *token.data;
    tokenData[Enums::Constr::IN] = tokenParameters._constructionInput;
    tokenData[Enums::Constr::IN_OPTION] = tokenParameters._constructionOption;
    tokenData[Enums::Constr::INOUT_ANGLE] = QuantityConverter::convertAngleToData(tokenParameters._angle);
    tokenData[Enums::Constr::IN_DIST] = QuantityConverter::convertDistanceToData(tokenParameters._distance);
    tokenData[Enums::Constr::IN_CELL_MAX_CONNECTIONS] = tokenParameters._maxConnections;
    tokenData[Enums::Constr::IN_CELL_BRANCH_NO] = tokenParameters._cellBranchNumber;
    tokenData[Enums::Constr::IN_CELL_FUNCTION_DATA] = tokenParameters._staticData.size();
    tokenData.replace(
        Enums::Constr::IN_CELL_FUNCTION_DATA + 1, tokenParameters._staticData.size(), tokenParameters._staticData);
    int const mutableDataIndex = Enums::Constr::IN_CELL_FUNCTION_DATA + 1 + tokenParameters._staticData.size();
    tokenData[mutableDataIndex] = tokenParameters._mutableData.size();
    tokenData.replace(mutableDataIndex + 1, tokenParameters._mutableData.size(), tokenParameters._mutableData);

    token.energy = tokenParameters._energy.get_value_or(
        2 * _parameters.tokenMinEnergy + 2 * _parameters.cellFunctionConstructorOffspringCellEnergy);
    return token;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getFirstCellOfConstructionSiteAfterCreation() const
{
    map<uint64_t, CellDescription> cellsBefore;
    for (auto const& cell : origConstructionSite) {
        cellsBefore.insert_or_assign(cell.id, cell);
    }
    map<uint64_t, CellDescription> cellsRemaining;
    for (auto const& cell : constructionSite) {
        if (cellsBefore.find(cell.id) == cellsBefore.end()) {
            cellsRemaining.insert_or_assign(cell.id, cell);
        }
    }

    CHECK(cellsRemaining.size() <= 1);
    if (!cellsRemaining.empty()) {
        return cellsRemaining.begin()->second;
    }

    return boost::none;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getFirstCellOfOrigConstructionSite() const
{
    if (!origConstructionSite.empty()) {
        return origConstructionSite.front();
    }
    return boost::none;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getSecondCellOfOrigConstructionSite() const
{
    if (origConstructionSite.size() > 1) {
        return origConstructionSite.at(1);
    }
    return boost::none;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getSecondCellOfConstructionSiteAfterCreation() const
{
    if (auto const firstCell = getFirstCellOfOrigConstructionSite()) {
        return getCellOfConstructionSite(firstCell->id);
    }
    return boost::none;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getThirdCellOfConstructionSiteAfterCreation() const
{
    if (auto const firstCell = getSecondCellOfOrigConstructionSite()) {
        return getCellOfConstructionSite(firstCell->id);
    }
    return boost::none;
}

optional<CellDescription> ConstructorGpuTests::TestResult::getCellOfConstructionSite(uint64_t id) const
{
    for (auto const& constructionSiteCell : constructionSite) {
        if (id == constructionSiteCell.id) {
            return constructionSiteCell;
        }
    }
    return boost::none;
}

void ConstructorGpuTests::_ResultChecker::check(TestResult const& testResult, Expectations const& expectations) const
{
    if (expectations._destruction) {
        checkIfDestruction(testResult, expectations);
    } else {
        checkIfNoDestruction(testResult, expectations);
    }
}

void ConstructorGpuTests::_ResultChecker::checkIfDestruction(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    auto const& token = testResult.token;

    EXPECT_EQ(expectations._tokenOutput, getConstrOut(token));

    if (Enums::ConstrIn::DO_NOTHING == getConstrIn(token)) {
        EXPECT_FALSE(testResult.getFirstCellOfConstructionSiteAfterCreation());
        return;
    }
}

void ConstructorGpuTests::_ResultChecker::checkIfNoDestruction(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    auto const& token = testResult.token;

    EXPECT_EQ(expectations._tokenOutput, getConstrOut(token));
    auto movementOfCenter = testResult.movementOfCenter;
    if (!isCompatible(movementOfCenter, QVector2D{})) {
        _spaceProp->correctPosition(movementOfCenter);
        EXPECT_TRUE(isCompatible(movementOfCenter, QVector2D{}));
    }
    
    checkTokenMovement(testResult);

    if (Enums::ConstrIn::DO_NOTHING == getConstrIn(token)) {
        EXPECT_FALSE(testResult.getFirstCellOfConstructionSiteAfterCreation());
        return;
    }

    if (Enums::ConstrOut::SUCCESS == expectations._tokenOutput) {
        checkIfNoDestructionAndSuccess(testResult, expectations);
    }
    else if (Enums::ConstrOut::SUCCESS_ROT== expectations._tokenOutput) {
        checkIfNoDestructionAndSuccessRotationOnly(testResult, expectations);
    } else {
        EXPECT_FALSE(testResult.getFirstCellOfConstructionSiteAfterCreation());
    }
}

void ConstructorGpuTests::_ResultChecker::checkIfNoDestructionAndSuccess(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    EXPECT_TRUE(testResult.getFirstCellOfConstructionSiteAfterCreation());
    EXPECT_EQ(1, testResult.constructionSite.size() - testResult.origConstructionSite.size());

    checkCellPositionAfterCreation(testResult, expectations);
    checkCellAttributesAfterCreation(testResult);
    checkCellConnectionsAfterCreation(testResult);
    checkConstructedTokenAfterCreation(testResult, expectations);
}

void ConstructorGpuTests::_ResultChecker::checkIfNoDestructionAndSuccessRotationOnly(TestResult const & testResult, Expectations const & expectations) const
{
    checkCellPositionAfterRotation(testResult, expectations);
    checkCellConnectionsAfterRotation(testResult, expectations);
}

void ConstructorGpuTests::_ResultChecker::checkCellPositionAfterCreation(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    if (testResult.origConstructionSite.empty()) {
        EXPECT_PRED3(
            predEqual,
            0,
            (*testResult.constructorCell.pos + *expectations._relPosOfFirstCellOfConstructionSite
             - *testResult.getFirstCellOfConstructionSiteAfterCreation()->pos)
                .length(),
            0.05);
    } else {

        //check distances
        checkConstructionSiteDistances(testResult);

        auto const origSourceCell = testResult.origSourceCell;
        auto const sourceCell = testResult.sourceCell;
        auto const firstCellOfConstructionSite = testResult.getFirstCellOfConstructionSiteAfterCreation();
        auto const firstCellOfOrigConstructionSite = *testResult.getFirstCellOfOrigConstructionSite();
        auto const secondCellOfOrigConstructionSite = testResult.getSecondCellOfOrigConstructionSite();
        auto const secondCellOfConstructionSite = *testResult.getSecondCellOfConstructionSiteAfterCreation();
        auto const expectedDistance =
            QuantityConverter::convertDataToDistance(testResult.origToken.data->at(Enums::Constr::IN_DIST));
        auto const expectedAngle =
            QuantityConverter::convertDataToAngle(testResult.origToken.data->at(Enums::Constr::INOUT_ANGLE));
        {
            auto const displacement = *secondCellOfConstructionSite.pos - *firstCellOfConstructionSite->pos;
            EXPECT_PRED3(predEqual, expectedDistance, displacement.length(), 0.05);
        }
        {
            auto const displacement = *firstCellOfConstructionSite->pos - *testResult.constructorCell.pos;
            if (isSeparated(testResult.origToken)) {
                EXPECT_PRED3(
                    predEqual, _parameters.cellFunctionConstructorOffspringCellDistance + expectedDistance, displacement.length(), 0.05);
            }
            else {
                EXPECT_PRED3(
                    predEqual, _parameters.cellFunctionConstructorOffspringCellDistance, displacement.length(), 0.05);
            }
        }
        if(sourceCell) {
            auto const displacementBefore = *origSourceCell.pos - *testResult.origConstructorCell.pos;
            auto const displacementAfter = *sourceCell->pos - *testResult.constructorCell.pos;
            EXPECT_PRED3(predEqual, displacementBefore.length(), displacementAfter.length(), 0.01);
        }

        //check angles
        if (testResult.sourceCell) {
            auto const origAngle = Physics::clockwiseAngleFromFirstToSecondVector(
                *firstCellOfOrigConstructionSite.pos - *testResult.origConstructorCell.pos,
                *testResult.origSourceCell.pos - *testResult.origConstructorCell.pos);
            auto const angle = Physics::clockwiseAngleFromFirstToSecondVector(
                *firstCellOfConstructionSite->pos - *testResult.constructorCell.pos,
                *testResult.sourceCell->pos - *testResult.constructorCell.pos);
            EXPECT_TRUE(isCompatible(origAngle, angle));
        }

        if (auto const thirdCellOfConstructionSite = testResult.getThirdCellOfConstructionSiteAfterCreation()) {
            auto const origAngle = Physics::clockwiseAngleFromFirstToSecondVector(
                *testResult.origConstructorCell.pos - *firstCellOfOrigConstructionSite.pos,
                *secondCellOfOrigConstructionSite->pos - *firstCellOfOrigConstructionSite.pos);
            auto const angle = Physics::clockwiseAngleFromFirstToSecondVector(
                *firstCellOfConstructionSite->pos - *secondCellOfConstructionSite.pos,
                *thirdCellOfConstructionSite->pos - *secondCellOfConstructionSite.pos);
            auto const actualDiffAngle = angle - origAngle;
            EXPECT_PRED3(predEqual, expectedAngle, actualDiffAngle, 1);
        }

        if (sourceCell) {
            auto const angularMasses = calcAngularMasses(testResult);
            auto const sumAngularMasses = angularMasses.constructor + angularMasses.constructionSite;
            auto const expectedDeltaAngleConstructionSite = angularMasses.constructor * expectedAngle / sumAngularMasses;
            auto const expectedDeltaAngleConstructor = -angularMasses.constructionSite * expectedAngle / sumAngularMasses;

            auto const origAngleConstructor =
                Physics::angleOfVector(*origSourceCell.pos - *testResult.origConstructorCell.pos);
            auto const angleConstructor =
                Physics::angleOfVector(*sourceCell->pos - *testResult.constructorCell.pos);
            EXPECT_PRED3(predEqual, expectedDeltaAngleConstructor, angleConstructor - origAngleConstructor, 0.001);

            if (testResult.origConstructionSite.size() >= 2) {
                auto const& origConstructionSiteCell1 = testResult.origConstructionSite.at(0);
                auto const& origConstructionSiteCell2 = testResult.origConstructionSite.at(1);
                auto const constructionSiteCell1 = testResult.getCellOfConstructionSite(origConstructionSiteCell1.id);
                auto const constructionSiteCell2 = testResult.getCellOfConstructionSite(origConstructionSiteCell2.id);
                auto const origAngleConstructionSite =
                    Physics::angleOfVector(*origConstructionSiteCell1.pos - *origConstructionSiteCell2.pos);
                auto const angleConstructionSite =
                    Physics::angleOfVector(*constructionSiteCell1->pos - *constructionSiteCell2->pos);

                EXPECT_PRED3(
                    predEqual,
                    expectedDeltaAngleConstructionSite,
                    angleConstructionSite - origAngleConstructionSite,
                    0.001);
            }
        }
    }
}

void ConstructorGpuTests::_ResultChecker::checkCellAttributesAfterCreation(TestResult const& testResult) const
{
    auto const constructedCell = *testResult.getFirstCellOfConstructionSiteAfterCreation();
    auto const& token = testResult.token;
    EXPECT_TRUE(isCompatible(
        _parameters.cellFunctionConstructorOffspringCellEnergy, static_cast<float>(*constructedCell.energy)));

    auto const expectedBranchNumber = token.data->at(Enums::Constr::IN_CELL_BRANCH_NO);
    auto const expectedCellFunctionType = token.data->at(Enums::Constr::IN_CELL_FUNCTION);

    auto const expectedStaticDataLength = token.data->at(Enums::Constr::IN_CELL_FUNCTION_DATA);
    auto const expectedStaticData = token.data->mid(Enums::Constr::IN_CELL_FUNCTION_DATA + 1, expectedStaticDataLength);
    auto const mutableDataIndex = Enums::Constr::IN_CELL_FUNCTION_DATA + 1 + expectedStaticDataLength;
    auto const expectedMutableDataLength = token.data->at(mutableDataIndex);
    auto const expectedMutableData = token.data->mid(mutableDataIndex + 1, expectedMutableDataLength);

    EXPECT_EQ(expectedBranchNumber, *constructedCell.tokenBranchNumber);
    EXPECT_EQ(expectedCellFunctionType, constructedCell.cellFeature->type);
    EXPECT_EQ(expectedStaticData, constructedCell.cellFeature->constData);
    EXPECT_EQ(expectedMutableData, constructedCell.cellFeature->volatileData);

    EXPECT_EQ(!isFinished(token), *constructedCell.tokenBlocked);
    if (auto const secondCell = testResult.getSecondCellOfConstructionSiteAfterCreation()) {
        EXPECT_FALSE(*secondCell->tokenBlocked);
    }
}

void ConstructorGpuTests::_ResultChecker::checkCellConnectionsAfterCreation(TestResult const& testResult) const
{
    auto const& token = testResult.token;
    auto const constructedCell = *testResult.getFirstCellOfConstructionSiteAfterCreation();
    EXPECT_EQ(!isSeparated(token), constructedCell.isConnectedTo(testResult.constructorCell.id));
    EXPECT_EQ(!isSeparated(token), testResult.constructorCell.isConnectedTo(constructedCell.id));

    for (auto const& cell : testResult.constructionSite) {
        EXPECT_TRUE(0 <= *cell.maxConnections);
        EXPECT_TRUE(*cell.maxConnections <= _parameters.cellMaxBonds);
    }

    for (auto const& origCell : testResult.origConstructionSite) {
        auto const cell = *testResult.getCellOfConstructionSite(origCell.id);
        for (auto const& connectingCellId : *origCell.connectingCells) {
            if (connectingCellId == testResult.origConstructorCell.id) {
                continue;
            }
            auto const connectingCell = *testResult.getCellOfConstructionSite(connectingCellId);
            EXPECT_TRUE(cell.isConnectedTo(connectingCell.id));
            EXPECT_TRUE(connectingCell.isConnectedTo(cell.id));
        }
    }

    auto const expectedMaxConnections = token.data->at(Enums::Constr::IN_CELL_MAX_CONNECTIONS);
    auto const isAutomaticMaxConnection = isAutomaticMaxConnections(token);
    if (isAutomaticMaxConnection) {
        auto const separatedAndNotReduced = isSeparated(token) && !isReduced(token) ? 1 : 0;
        EXPECT_EQ(constructedCell.connectingCells->size() + separatedAndNotReduced, *constructedCell.maxConnections);
    }
    else {
        EXPECT_EQ(expectedMaxConnections, *constructedCell.maxConnections);
    }
    for (auto const& cell : testResult.constructionSite) {
        if (cell.id == constructedCell.id) {
            continue;
        }
        if ((*constructedCell.pos - *cell.pos).length() < _parameters.cellMaxDistance) {
            if (isAutomaticMaxConnection) {
                if (*constructedCell.maxConnections < _parameters.cellMaxBonds
                    && *cell.maxConnections < _parameters.cellMaxBonds) {
                    EXPECT_TRUE(cell.isConnectedTo(constructedCell.id));
                    EXPECT_TRUE(constructedCell.isConnectedTo(cell.id));
                }
            }
            else {
                if (constructedCell.connectingCells->size() < *constructedCell.maxConnections
                    && cell.connectingCells->size() < *cell.maxConnections) {
                    EXPECT_TRUE(cell.isConnectedTo(constructedCell.id));
                    EXPECT_TRUE(constructedCell.isConnectedTo(cell.id));
                }
            }
        }
        else {
            EXPECT_FALSE(cell.isConnectedTo(constructedCell.id));
            EXPECT_FALSE(constructedCell.isConnectedTo(cell.id));
        }
    }

    auto const increaseMaxConnectionIfNewConstructionSite = 0 == testResult.origConstructionSite.size() ? 1 : 0;
    if (*testResult.origConstructorCell.maxConnections == testResult.origConstructorCell.connectingCells->size()) {
        auto const decreaseMaxConnectionIfReduced = isReduced(token) ? -1 : 0;
        EXPECT_EQ(
            *testResult.origConstructorCell.maxConnections + increaseMaxConnectionIfNewConstructionSite
                + decreaseMaxConnectionIfReduced,
            *testResult.constructorCell.maxConnections);
    }
}

void ConstructorGpuTests::_ResultChecker::checkConstructedTokenAfterCreation(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    if (expectations._constructedToken) {
        auto const actualTokens = testResult.getFirstCellOfConstructionSiteAfterCreation()->tokens;
        EXPECT_EQ(1, actualTokens->size());
        EXPECT_TRUE(isCompatible(*expectations._constructedToken, actualTokens->at(0)));

        EXPECT_EQ(1, testResult.getFirstCellOfConstructionSiteAfterCreation()->tokens->size());
    }
    else {
        EXPECT_TRUE(testResult.getFirstCellOfConstructionSiteAfterCreation()->tokens->empty());
    }
}

void ConstructorGpuTests::_ResultChecker::checkCellPositionAfterRotation(TestResult const & testResult, Expectations const & expectations) const
{
    //check distances
    checkConstructionSiteDistances(testResult);
    auto const origSourceCell = testResult.origSourceCell;
    auto const sourceCell = testResult.sourceCell;
    auto const origConstructor = testResult.origConstructorCell;
    auto const constructor = testResult.constructorCell;
    auto const firstCellOfOrigConstructionSite = *testResult.getFirstCellOfOrigConstructionSite();
    auto const firstCellOfConstructionSite = *testResult.getCellOfConstructionSite(firstCellOfOrigConstructionSite.id);
    auto const secondCellOfOrigConstructionSite = *testResult.getSecondCellOfOrigConstructionSite();
    auto const secondCellOfConstructionSite = *testResult.getCellOfConstructionSite(secondCellOfOrigConstructionSite.id);
    {
        auto const displacementBefore = *origConstructor.pos - *firstCellOfOrigConstructionSite.pos;
        auto const displacementAfter = *constructor.pos - *firstCellOfConstructionSite.pos;
        EXPECT_PRED3(predEqual, displacementBefore.length(), displacementAfter.length(), 0.01);
    }
    {
        auto const displacementBefore = *origSourceCell.pos - *testResult.origConstructorCell.pos;
        auto const displacementAfter = *sourceCell->pos - *testResult.constructorCell.pos;
        EXPECT_PRED3(predEqual, displacementBefore.length(), displacementAfter.length(), 0.01);
    }

    auto const expectedAngleDelta =
        QuantityConverter::convertDataToAngle(testResult.origToken.data->at(Enums::Constr::INOUT_ANGLE))
        - QuantityConverter::convertDataToAngle(testResult.token.data->at(Enums::Constr::INOUT_ANGLE));

    auto const origAngle = Physics::clockwiseAngleFromFirstToSecondVector(
        *secondCellOfOrigConstructionSite.pos - *firstCellOfOrigConstructionSite.pos,
        *origConstructor.pos - *firstCellOfOrigConstructionSite.pos);
    auto const angle = Physics::clockwiseAngleFromFirstToSecondVector(
        *secondCellOfConstructionSite.pos - *firstCellOfConstructionSite.pos,
        *constructor.pos - *firstCellOfConstructionSite.pos);
    EXPECT_PRED3(predEqual, expectedAngleDelta, origAngle - angle, 0.001);

}

void ConstructorGpuTests::_ResultChecker::checkCellConnectionsAfterRotation(
    TestResult const& testResult,
    Expectations const& expectations) const
{
    for (auto const& origCell : testResult.origConstructionSite) {
        auto const cell = *testResult.getCellOfConstructionSite(origCell.id);
        EXPECT_EQ(origCell.connectingCells, cell.connectingCells);
    }
    EXPECT_EQ(testResult.origSourceCell.connectingCells, testResult.sourceCell->connectingCells);
    EXPECT_EQ(testResult.origConstructorCell.connectingCells, testResult.constructorCell.connectingCells);
}

void ConstructorGpuTests::_ResultChecker::checkTokenMovement(TestResult const & testResult) const
{
    unordered_map<uint64_t, CellDescription> cellById;
    for (auto const& cell : testResult.origConstructionSite) {
        cellById.insert_or_assign(cell.id, cell);
    }

    unordered_map<uint64_t, int> expectedNumTokensByCellId;
    for (auto const& cell : testResult.origConstructionSite) {
        for (auto const& connectingCellId : *cell.connectingCells) {
            auto const connectingCellFindResult = cellById.find(connectingCellId);
            if (connectingCellFindResult == cellById.end()) {
                continue;
            }
            auto const& connectingCell = connectingCellFindResult->second;
            if ((*cell.tokenBranchNumber + 1) % _parameters.cellMaxTokenBranchNumber
                == *connectingCell.tokenBranchNumber) {
                ++expectedNumTokensByCellId[connectingCellId];
            }
        }
    }

    for (auto const& cell : testResult.origConstructionSite) {
        auto const expectedNumTokensFindResult = expectedNumTokensByCellId.find(cell.id);
        if (expectedNumTokensFindResult != expectedNumTokensByCellId.end()) {
            EXPECT_EQ(expectedNumTokensFindResult->second, cell.tokens->size());
        }
    }
}

void ConstructorGpuTests::_ResultChecker::checkConstructionSiteDistances(TestResult const & testResult) const
{
    optional<CellDescription> prevOrigCell;
    for (auto const& origCell : testResult.origConstructionSite) {
        if (prevOrigCell) {
            auto const prevCell = *testResult.getCellOfConstructionSite(prevOrigCell->id);
            auto const cell = *testResult.getCellOfConstructionSite(origCell.id);

            EXPECT_PRED3(
                predEqual, (*cell.pos - *prevCell.pos).length(), (*origCell.pos - *prevOrigCell->pos).length(), 0.01);
        }
        prevOrigCell = origCell;
    }
}

auto ConstructorGpuTests::_ResultChecker::calcAngularMasses(TestResult const & testResult) const -> AngularMasses
{
    AngularMasses result;

    auto const firstCellOfOrigConstructionSite = *testResult.getFirstCellOfOrigConstructionSite();
    vector<QVector2D> relPositionOfMasses;
    std::transform(
        testResult.origConstructor.begin(),
        testResult.origConstructor.end(),
        std::inserter(relPositionOfMasses, relPositionOfMasses.begin()),
        [&firstCellOfOrigConstructionSite](auto const& cell) {
        return *cell.pos - *firstCellOfOrigConstructionSite.pos;
    });
    result.constructor = Physics::angularMass(relPositionOfMasses);

    relPositionOfMasses.clear();
    std::transform(
        testResult.origConstructionSite.begin(),
        testResult.origConstructionSite.end(),
        std::inserter(relPositionOfMasses, relPositionOfMasses.begin()),
        [&firstCellOfOrigConstructionSite](auto const& cell) {
        return *cell.pos - *firstCellOfOrigConstructionSite.pos;
    });
    result.constructionSite = Physics::angularMass(relPositionOfMasses);

    return result;
}

Enums::ConstrIn::Type ConstructorGpuTests::_ResultChecker::getConstrIn(TokenDescription const & token) const
{
    return static_cast<Enums::ConstrIn::Type>(
        static_cast<unsigned char>(token.data->at(Enums::Constr::IN)) % Enums::ConstrIn::_COUNTER);
}

Enums::ConstrOut::Type ConstructorGpuTests::_ResultChecker::getConstrOut(TokenDescription const & token) const
{
    return static_cast<Enums::ConstrOut::Type>(token.data->at(Enums::Constr::OUT));
}

bool ConstructorGpuTests::isFinished(TokenDescription const& token)
{
    auto const option = static_cast<unsigned char>(token.data->at(Enums::Constr::IN_OPTION) % Enums::ConstrInOption::_COUNTER);
    return Enums::ConstrInOption::FINISH_NO_SEP == option || Enums::ConstrInOption::FINISH_WITH_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;
}

bool ConstructorGpuTests::isReduced(TokenDescription const& token)
{
    auto const option = static_cast<unsigned char>(token.data->at(Enums::Constr::IN_OPTION) % Enums::ConstrInOption::_COUNTER);
    return Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;
}

bool ConstructorGpuTests::isSeparated(TokenDescription const& token)
{
    auto const option = static_cast<unsigned char>(token.data->at(Enums::Constr::IN_OPTION) % Enums::ConstrInOption::_COUNTER);
    return Enums::ConstrInOption::FINISH_WITH_SEP == option || Enums::ConstrInOption::FINISH_WITH_SEP_RED == option
        || Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED == option;
}

bool ConstructorGpuTests::isAutomaticMaxConnections(TokenDescription const & token)
{
    auto const maxConnections = token.data->at(Enums::Constr::IN_CELL_MAX_CONNECTIONS);
    return 0 == maxConnections;
}

QVector2D ConstructorGpuTests::constructorPositionForHorizontalClusterAfterCreation(
    vector<QVector2D> constructionSite,
    QVector2D constructor,
    vector<QVector2D> remainingCells,
    float distanceBetweenOffspringToConstructionSite,
    WithSeparation withSeparation) const
{
    auto origCenter = std::accumulate(constructionSite.begin(), constructionSite.end(), constructor);
    origCenter = std::accumulate(remainingCells.begin(), remainingCells.end(), origCenter);
    origCenter /= constructionSite.size() + 1 + remainingCells.size();

    auto offspringCellPos = constructor + QVector2D{_offspringDistance, 0};
    if (withSeparation == WithSeparation::Yes) {
        offspringCellPos += QVector2D{distanceBetweenOffspringToConstructionSite, 0};
    }
    for (auto& pos : constructionSite) {
        pos += QVector2D{distanceBetweenOffspringToConstructionSite, 0};
    }

    auto center = std::accumulate(constructionSite.begin(), constructionSite.end(), constructor + offspringCellPos);
    center = std::accumulate(remainingCells.begin(), remainingCells.end(), center);
    center /= constructionSite.size() + 2 + remainingCells.size();

    auto const centerDisplacement = center - origCenter;
    return constructor - centerDisplacement;
}

TEST_F(ConstructorGpuTests, testDoNothing)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::DO_NOTHING));
    auto result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_standardParameters)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_nonStandardParameters1)
{
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .cellBranchNumber(2)
                                                      .maxConnections(3)
                                                      .cellFunctionType(Enums::CellFunction::SCANNER));
    auto result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_nonStandardParameters2)
{
    auto const basicFacade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
    auto const compiler =
        basicFacade->buildCellComputerCompiler(_context->getSymbolTable(), _context->getSimulationParameters());

    std::stringstream stream;
    stream << "mov [1], 3";
    for (int i = 0; i < _parameters.cellFunctionComputerMaxInstructions - 1; ++i) {
        stream << "\nmov [1], 3";
    }

    CompilationResult compiledProgram = compiler->compileSourceCode(stream.str());
    CHECK(compiledProgram.compilationOk);

    auto const token =
        createTokenForConstruction(TokenForConstructionParameters()
                                       .constructionInput(Enums::ConstrIn::SAFE)
                                       .cellBranchNumber(1)
                                       .maxConnections(2)
                                       .cellFunctionType(Enums::CellFunction::COMPUTER)
                                       .staticData(compiledProgram.compilation)
                                       .mutableData(QByteArray(_parameters.cellFunctionComputerCellMemorySize, 1)));
    auto result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_ignoreDistanceOnFirstConstructedCell1)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(_offspringDistance / 2));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_ignoreDistanceOnFirstConstructedCell2)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(_offspringDistance * 2));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_rightHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90.0f));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_leftHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(-90.0f));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{0, -_offspringDistance};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_setAutomaticMaxConnectionsOnConstructor)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).maxConnectionsOfConstructor(1));

    auto const expectedCellPos = QVector2D{ _offspringDistance, 0 };
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_errorMaxConnections1)
{
    _parameters.cellMaxBonds = 1;
    _context->setSimulationParameters(_parameters);

    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).maxConnectionsOfConstructor(1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_errorMaxConnections2)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).maxConnections(1));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).maxConnectionsOfConstructor(1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_errorNoEnergy)
{
    auto const lowTokenEnergy = _parameters.tokenMinEnergy + _parameters.cellFunctionConstructorOffspringCellEnergy / 2;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).energy(lowTokenEnergy));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_NO_ENERGY));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterRightObstacle_safe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(_offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterRightObstacle_unsafe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(_offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterRightObstacle_bruteforce)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(_offspringDistance));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterLeftObstacle_safe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            -1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterLeftObstacle_unsafe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            -1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_otherClusterLeftObstacle_bruteforce)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            -1));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_ownClusterObstacle_safe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(180));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_ownClusterObstacle_unsafe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE).angle(90));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(180));

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_ownClusterObstacle_bruteforce)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE).angle(90));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(180));

    auto const expectedCellPos = QVector2D{0, _offspringDistance};
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_ownClusterObstacle_backward)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(180.0f));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_rightHandSide)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(90));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_leftHandSide)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(270));

    auto const expectedCellPos = QVector2D{-_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_diagonal)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(90).clusterAngle(45));

    auto const expectedCellPos = QVector2D{_offspringDistance / sqrtf(2), _offspringDistance / sqrtf(2)};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnWedgeCluster_beyondUniverseBoundaries)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnWedgeClusterTest(
        StartConstructionOnWedgeClusterTestParameters().token(token).wedgeAngle(90).clusterAngle(45).referencePosition(
            QVector2D{0, 0}));

    auto const expectedCellPos = QVector2D{ _offspringDistance / sqrtf(2), _offspringDistance / sqrtf(2) };
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnTiangleCluster)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result =
        runStartConstructionOnTriangleClusterTest(StartConstructionOnTriangleClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_withEmptyToken)
{
    auto const cellBranchNumber = 1;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                                                      .cellBranchNumber(cellBranchNumber));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
                                   .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                                   .setData(expectedTokenMemory);
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_withDuplicatedToken)
{
    auto const cellBranchNumber = 1;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::CREATE_DUP_TOKEN)
                                                      .cellBranchNumber(cellBranchNumber));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    auto expectedTokenMemory = *token.data;
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
                                   .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                                   .setData(expectedTokenMemory);
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_finishWithoutSeparation)
{
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_NO_SEP));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{_offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_finishWithSeparation)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{cellDistance + _offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(
    ConstructorGpuTests,
    testConstructFirstCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_safe)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            cellDistance + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(
    ConstructorGpuTests,
    testConstructFirstCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_unsafe)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::UNSAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            cellDistance + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(
    ConstructorGpuTests,
    testConstructFirstCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_bruteforce)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::BRUTEFORCE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runStartConstructionOnHorizontalClusterTest(
        StartConstructionOnHorizontalClusterTestParameters().token(token).horizontalObstacleAt(
            cellDistance + _offspringDistance));

    auto const expectedCellPos = QVector2D{1 + _offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_finishWithSeparationAndReduction)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                                                      .distance(cellDistance));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    auto const expectedCellPos = QVector2D{cellDistance + _offspringDistance, 0};
    _resultChecker->check(
        result,
        Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).relPosOfFirstCellOfConstructionSite(expectedCellPos));
}

TEST_F(ConstructorGpuTests, testConstructFirstCellOnHorizontalCluster_finishWithTokenAndSeparationAndReduction)
{
    auto const cellBranchNumber = 1;
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters()
                                       .constructionInput(Enums::ConstrIn::SAFE)
                                       .constructionOption(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)
                                       .cellBranchNumber(cellBranchNumber)
                                       .distance(cellDistance));
    auto const result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));

    auto const expectedCellPos = QVector2D{cellDistance + _offspringDistance, 0};
    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
                                   .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                                   .setData(expectedTokenMemory);
    _resultChecker->check(
        result,
        Expectations()
            .tokenOutput(Enums::ConstrOut::SUCCESS)
            .relPosOfFirstCellOfConstructionSite(expectedCellPos)
            .constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_standardParameters)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_minDistance)
{
    auto const minDistance = _parameters.cellMinDistance;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(minDistance * 1.1f));
    auto result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_maxDistance)
{
    auto const maxDistance = _parameters.cellMaxDistance;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(maxDistance * 0.9f));
    auto result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}


TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_rightHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_leftHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(-30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnLShapeCluster_rightHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).angleOfConstructionSite(90));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnLShapeCluster_leftHandSide)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(-30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).angleOfConstructionSite(90));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_errorTooLowDistance)
{
    auto const minDistance = _parameters.cellMinDistance;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(minDistance * 0.9f));
    auto result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_DIST));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_errorTooLargeDistance)
{
    auto const maxDistance = _parameters.cellMaxDistance;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).distance(maxDistance * 1.1f));
    auto result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_DIST));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_errorNoEnergy)
{
    auto const lowTokenEnergy = _parameters.tokenMinEnergy + _parameters.cellFunctionConstructorOffspringCellEnergy / 2;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).energy(lowTokenEnergy));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_NO_ENERGY));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_errorMaxConnections1)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).maxConnections(1));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_errorMaxConnections2)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).maxConnections(8));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterRightObstacle_safe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            1 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterRightObstacle_unsafe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            1 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterRightObstacle_bruteforce)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            1 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).destruction(true));
}


TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterLeftObstacle_safe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(-1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterLeftObstacle_unsafe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(-1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_otherClusterLeftObstacle_bruteforce)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(-1));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnSelfTouchingCluster_ownClusterObstacle_safe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runSecondCellConstructionOnSelfTouchingClusterTest(token, 4);

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnSelfTouchingCluster_ownClusterObstacle_unsafe)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE));
    auto const result = runSecondCellConstructionOnSelfTouchingClusterTest(token, 4);

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnSelfTouchingCluster_ownClusterObstacle_bruteforce)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE));
    auto const result = runSecondCellConstructionOnSelfTouchingClusterTest(token, 4);

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_ownClusterObstacle_backward)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(180.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));   //no rotation takes place
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_withEmptyToken)
{
    auto const cellBranchNumber = 1;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
        .constructionInput(Enums::ConstrIn::SAFE)
        .constructionOption(Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
        .cellBranchNumber(cellBranchNumber));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
        .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
        .setData(expectedTokenMemory);
    _resultChecker->check(
        result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_withEmptyToken_tokenAnreadyOnConstructionSite)
{
    auto const cellBranchNumber = 2;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::CREATE_EMPTY_TOKEN)
                                                      .cellBranchNumber(cellBranchNumber));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).tokenOnConstructionSite(
            createSimpleToken()));
    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
                                   .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                                   .setData(expectedTokenMemory);
    _resultChecker->check(
        result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_withDuplicatedToken)
{
    auto const cellBranchNumber = 1;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::CREATE_DUP_TOKEN)
                                                      .cellBranchNumber(cellBranchNumber));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    auto expectedTokenMemory = *token.data;
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
                                   .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
                                   .setData(expectedTokenMemory);
    _resultChecker->check(
        result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedToken(expectedToken));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_finishWithoutSeparation)
{
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_NO_SEP));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_finishWithSeparation_standardPosition)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_finishWithSeparation_nonStandardPosition)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance)
                                                      .angle(30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnLShapeCluster_finishWithSeparation_nonStandardPosition)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
        .constructionInput(Enums::ConstrIn::SAFE)
        .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
        .distance(cellDistance)
        .angle(30.0f));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).angleOfConstructionSite(90.0f));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_finishWithSeparationAndReduction)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP_RED)
                                                      .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}


TEST_F(ConstructorGpuTests, testConstructSecondCellOnHorizontalCluster_finishWithTokenAndSeparationAndReduction)
{
    auto const cellBranchNumber = 1;
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters()
                                       .constructionInput(Enums::ConstrIn::SAFE)
                                       .constructionOption(Enums::ConstrInOption::FINISH_WITH_TOKEN_SEP_RED)
                                       .cellBranchNumber(cellBranchNumber)
                                       .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));

    QByteArray expectedTokenMemory(_parameters.tokenMemorySize, 0);
    expectedTokenMemory[0] = cellBranchNumber;
    auto const expectedToken = TokenDescription()
        .setEnergy(_parameters.cellFunctionConstructorOffspringTokenEnergy)
        .setData(expectedTokenMemory);
    _resultChecker->check(
        result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).constructedToken(expectedToken));
}

TEST_F(
    ConstructorGpuTests,
    testConstructSecondCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_safe)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::SAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            cellDistance*2 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(
    ConstructorGpuTests,
    testConstructSecondCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_unsafe)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::UNSAFE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            cellDistance * 2 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(
    ConstructorGpuTests,
    testConstructSecondCellOnHorizontalCluster_finishWithSeparation_otherClusterRightObstacle_bruteforce)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(TokenForConstructionParameters()
                                                      .constructionInput(Enums::ConstrIn::BRUTEFORCE)
                                                      .constructionOption(Enums::ConstrInOption::FINISH_WITH_SEP)
                                                      .distance(cellDistance));
    auto const result = runSecondConstructionOnLineClusterTest(
        SecondCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).horizontalObstacleAt(
            cellDistance * 2 + _offspringDistance));

    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructThirdCellOnHorizontalCluster_standardParameters)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructThirdCellOnLineCluster_nonStandardParameters)
{
    auto const cellDistance = _parameters.cellMinDistance * 1.1f;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(10).distance(cellDistance));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            {{90, 0}, {180, 0}}));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructThirdCellOnLineCluster_tokenOnConstructionSite)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            {{90, 0, boost::none, 0}, {180, 0, createSimpleToken(), 1}}));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testConstructThirdCellOnLineCluster_noMultipleConnections_limitMaxConnections)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(0).maxConnections(2));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            {{90, 0}, {180, 0}}));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_otherClusterObstacle_safe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).verticalObstacleAt(-0.5));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_otherClusterObstacle_unsafe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE).angle(120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).verticalObstacleAt(-0.5));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_otherClusterObstacle_bruteforce)
{
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE).angle(120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).verticalObstacleAt(-0.5));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT).destruction(true));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_ownClusterObstacle_safe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(-120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            {{60, 0}, {90, 0}, {180, 0}, {180, 0}}));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_OBSTACLE));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_ownClusterObstacle_unsafe)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::UNSAFE).angle(-120));
    auto result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
    { { 60, 0 },{ 90, 0 },{ 180, 0 },{ 180, 0 } }));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT).destruction(true));
}

TEST_F(ConstructorGpuTests, testRotationOnlyOnHorizontalCluster_ownClusterObstacle_bruteforce)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::BRUTEFORCE).angle(-120));
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
    { { 60, 0 },{ 90, 0 },{ 180, 0 },{ 180, 0 } }));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT).destruction(true));
}

TEST_F(ConstructorGpuTests, testConstructionSiteConnectedToConstructorTwice_errorConnection)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runConstructionSiteConnectedToConstructorTwiceTest(token);
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_CONNECTION));
}

TEST_F(ConstructorGpuTests, testLargeCluster_limitSize)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));

    int minSize = std::min(_universeSize.x, _universeSize.y);
    CHECK(0 == minSize % 4);
    int maxRadius = minSize / 2 - 1;

    vector<FurtherCellConstructionOnLineClusterTestParameters::CellProperties> constructionSiteProperties;
    for (int i = 0; i < maxRadius - 1; ++i) {
        constructionSiteProperties.emplace_back(
            FurtherCellConstructionOnLineClusterTestParameters::CellProperties{180.0f, 0});
    }
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            constructionSiteProperties));
    EXPECT_NE(Enums::ConstrOut::ERROR_MAX_RADIUS, result.token.data->at(Enums::Constr::OUT));
}

TEST_F(ConstructorGpuTests, testLargeCluster_errorMaxRadius)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));

    int minSize = std::min(_universeSize.x, _universeSize.y);
    CHECK(0 == minSize % 4);
    int maxRadius = minSize / 2;

    vector<FurtherCellConstructionOnLineClusterTestParameters::CellProperties> constructionSiteProperties;
    for (int i = 0; i < maxRadius - 1; ++i) {
        constructionSiteProperties.emplace_back(
            FurtherCellConstructionOnLineClusterTestParameters::CellProperties{ 180.0f, 0 });
    }
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            constructionSiteProperties));
    EXPECT_EQ(Enums::ConstrOut::ERROR_MAX_RADIUS, result.token.data->at(Enums::Constr::OUT));
}

TEST_F(ConstructorGpuTests, testParallelConstructionFromDifferentSources)
{
    auto const maxToken = _parameters.cellMaxToken;
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));
    auto const result = runStartConstructionOnTriangleClusterTest(StartConstructionOnTriangleClusterTestParameters()
                                                                      .token(token)
                                                                      .tokensOnSource1(maxToken / 2)
                                                                      .tokensOnSource2(maxToken / 2)
                                                                      .tokensOnSource3(maxToken / 2));

    for (auto const& token : *result.constructorCell.tokens) {
        EXPECT_EQ(Enums::ConstrOut::SUCCESS, token.data->at(Enums::Constr::OUT));
    }
    EXPECT_EQ(maxToken, result.constructionSite.size() - result.origConstructionSite.size());
}

TEST_F(ConstructorGpuTests, testParallelConstructionFromDifferentConstructors_manyIsolatedClusters)
{
    auto testResult = runMassiveParallelClustersTest(
        MassiveParallelClustersTestParameters().clusterLen(100).numClusters(10).distanceBetweenClusters(6));
    EXPECT_EQ(10, testResult.numCellsPerCluster.size());
    for (auto const& clusterSize : testResult.numCellsPerCluster) {
        EXPECT_EQ(100 * 4, clusterSize);
    }
}

TEST_F(ConstructorGpuWithHighBlockCountTests, testParallelConstructionFromDifferentConstructors_greatManyIsolatedClusters)
{
    auto testResult = runMassiveParallelClustersTest(
        MassiveParallelClustersTestParameters().clusterLen(4).numClusters(64 * 8 * 4).distanceBetweenClusters(6));
    EXPECT_EQ(64 * 8 * 4, testResult.numCellsPerCluster.size());
    for (auto const& clusterSize : testResult.numCellsPerCluster) {
        EXPECT_EQ(4 * 4, clusterSize);
    }
}

TEST_F(ConstructorGpuTests, testParallelConstructionFromDifferentConstructors_touchingClusters)
{
    EXPECT_NO_THROW(runMassiveParallelClustersTest(
        MassiveParallelClustersTestParameters().clusterLen(100).numClusters(10).distanceBetweenClusters(3)));
}

TEST_F(ConstructorGpuTests, testMultipleConnectedConstructionSites_errorConnection)
{
    DataDescription origData;
    auto cluster = createRectangularCluster({3, 2}, QVector2D{10.5, 10.5}, QVector2D{});

    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE));

    for(int i = 0; i <= 3; i += 3) {
        {
            auto& cell = cluster.cells->at(i);
            cell.tokenBranchNumber = 0;
            cell.addToken(token);
        }
        {
            auto& cell = cluster.cells->at(i + 1);
            cell.tokenBranchNumber = 1;
            cell.cellFeature = CellFeatureDescription().setType(Enums::CellFunction::CONSTRUCTOR);
        }
        {
            auto& cell = cluster.cells->at(i + 2);
            cell.tokenBlocked = true;
        }
    }
    origData.addCluster(cluster);

    IntegrationTestHelper::updateData(_access, origData);

    //perform test
    IntegrationTestHelper::runSimulation(1, _controller);

    //check results
    DataDescription newData = IntegrationTestHelper::getContent(_access, { { 0, 0 },{ _universeSize.x, _universeSize.y } });
    check(origData, newData);

    EXPECT_EQ(1, newData.clusters->size());

    auto const& newCluster = newData.clusters->front();
    EXPECT_EQ(6, newCluster.cells->size());

    for (int i = 0; i <= 3; i += 3) {
        auto const& token = newCluster.cells->at(i + 1).tokens->at(0);
        EXPECT_EQ(Enums::ConstrOut::ERROR_CONNECTION, token.data->at(Enums::Constr::OUT));
    }
}

TEST_F(ConstructorGpuTests, testLargeConstructionSite)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90));

    vector<FurtherCellConstructionOnLineClusterTestParameters::CellProperties> constructionSiteProperties;
    for (int i = 0; i < 140; ++i) {
        constructionSiteProperties.emplace_back(
            FurtherCellConstructionOnLineClusterTestParameters::CellProperties{ 180.0f, 0 });
    }
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            constructionSiteProperties));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT));
}

TEST_F(ConstructorGpuTests, testLargeConstructor)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90));

    vector<FurtherCellConstructionOnLineClusterTestParameters::CellProperties> constructionSiteProperties;
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).additionalCellsOnConstructor(140));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS_ROT));
}

TEST_F(ConstructorGpuTests, testLargeConstructionSiteAndConstructor)
{
    auto const token =
        createTokenForConstruction(TokenForConstructionParameters().constructionInput(Enums::ConstrIn::SAFE).angle(90));

    vector<FurtherCellConstructionOnLineClusterTestParameters::CellProperties> constructionSiteProperties;
    for (int i = 0; i < 140; ++i) {
        constructionSiteProperties.emplace_back(
            FurtherCellConstructionOnLineClusterTestParameters::CellProperties{ 180.0f, 0 });
    }
    auto const result = runFurtherCellConstructionOnLineClusterTest(
        FurtherCellConstructionOnLineClusterTestParameters().tokenOnSourceCell(token).propertiesOfConstructionSite(
            constructionSiteProperties).additionalCellsOnConstructor(140));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::ERROR_DIST));
}

TEST_F(ConstructorGpuTests, regressionTestNegativeValueForCommand)
{
    int command = Enums::ConstrIn::DO_NOTHING;
    command += (256 / Enums::ConstrIn::_COUNTER - 1) * Enums::ConstrIn::_COUNTER;
    auto const token = createTokenForConstruction(
        TokenForConstructionParameters().constructionInput(static_cast<Enums::ConstrIn::Type>(command)));
    auto result =
        runStartConstructionOnHorizontalClusterTest(StartConstructionOnHorizontalClusterTestParameters().token(token));
    _resultChecker->check(result, Expectations().tokenOutput(Enums::ConstrOut::SUCCESS));
}
