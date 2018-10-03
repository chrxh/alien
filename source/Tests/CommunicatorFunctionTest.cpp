#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"

#include "ModelInterface/ModelBasicBuilderFacade.h"
#include "ModelInterface/Settings.h"
#include "ModelInterface/SimulationParameters.h"
#include "ModelInterface/SimulationController.h"
#include "ModelInterface/SimulationAccess.h"
#include "ModelCpu/Cluster.h"
#include "ModelCpu/Cell.h"
#include "ModelCpu/Token.h"
#include "ModelCpu/PhysicalQuantityConverter.h"
#include "ModelCpu/UnitContext.h"
#include "ModelCpu/CommunicatorFunction.h"

#include "Tests/TestSettings.h"

#include "IntegrationTestFramework.h"
#include "IntegrationTestHelper.h"

class CommunicatorFunctionTest : public IntegrationTestFramework
{
public:
	CommunicatorFunctionTest();
	~CommunicatorFunctionTest();

protected:
    SimulationController* _controller = nullptr;
	SimulationAccess* _access = nullptr;
};

CommunicatorFunctionTest::CommunicatorFunctionTest()
	: IntegrationTestFramework({ 1000, 1000 })
{
	ModelBasicBuilderFacade* facade = ServiceLocator::getInstance().getService<ModelBasicBuilderFacade>();
	_controller = facade->buildSimulationController(1, { 1, 1 }, _universeSize, _symbols, _parameters);
	auto context = _controller->getContext();

	_access = facade->buildSimulationAccess();
	_access->init(context);
}

TEST_F(CommunicatorFunctionTest, testSendMessage_receiveOnSameChannel)
{
	const float distCommRange = _parameters->cellFunctionCommunicatorRange / 2.0;
	const uint8_t channel = 1;
	const uint8_t differentChannel = 2;
	const uint8_t message = 100;
	const uint8_t angle = PhysicalQuantityConverter::convertAngleToData(180.0);

	QByteArray tokenData(_parameters->tokenMemorySize, 0);
	tokenData[Enums::Communicator::IN] = Enums::CommunicatorIn::SEND_MESSAGE;
	tokenData[Enums::Communicator::IN_CHANNEL] = channel;
	tokenData[Enums::Communicator::IN_MESSAGE] = message;
	tokenData[Enums::Communicator::IN_ANGLE] = angle;
	tokenData[Enums::Communicator::IN_DISTANCE] = static_cast<uint8_t>(distCommRange);

	QByteArray cellData(5, 0);
	cellData[CommunicatorFunction::InternalDataSemantic::Channel] = channel;

	const uint64_t clusterId1 = 1;
	const uint64_t clusterId2 = 2;
	const uint64_t cellId1 = 3;
	const uint64_t cellId2 = 4;
	const uint64_t cellId3 = 5;
	auto dataInit = DataDescription().addClusters({
		ClusterDescription()
		.setId(clusterId1)
		.addCells({
			CellDescription()
			.setId(cellId1).setPos({ 500, 500 }).setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy).setFlagTokenBlocked(false)
			.setConnectingCells({ cellId2 }).setMaxConnections(1).setTokenBranchNumber(0)
			.setCellFeature(
				CellFeatureDescription().setType(Enums::CellFunction::COMMUNICATOR).setVolatileData(cellData)
			)
			.setTokens({
				TokenDescription().setEnergy(_parameters->cellFunctionConstructorOffspringTokenEnergy).setData(tokenData)
			}),
			CellDescription()
			.setId(cellId2).setPos({ 500, 501 }).setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy).setFlagTokenBlocked(false)
			.setConnectingCells({ cellId1 }).setMaxConnections(1).setTokenBranchNumber(1)
			.setCellFeature(
				CellFeatureDescription().setType(Enums::CellFunction::COMMUNICATOR).setVolatileData(cellData)
			)
		}),
		ClusterDescription()
		.setId(clusterId2)
		.addCells({
			CellDescription()
			.setId(cellId3).setPos({ 500 + distCommRange / 2, 501 + distCommRange / 2 }).setEnergy(_parameters->cellFunctionConstructorOffspringCellEnergy)
			.setFlagTokenBlocked(false).setMaxConnections(0).setTokenBranchNumber(0)
			.setCellFeature(
				CellFeatureDescription().setType(Enums::CellFunction::COMMUNICATOR).setVolatileData(cellData)
			)
		})
	});
	_access->updateData(dataInit);

	runSimulation(1, _controller);

	IntRect rect = { { 0, 0 },{ _universeSize.x - 1, _universeSize.y - 1 } };
	DataDescription dataAfter = IntegrationTestHelper::getContent(_access, rect);
	unordered_map<uint64_t, CellDescription> cellById = IntegrationTestHelper::getCellById(dataAfter);

	QByteArray cellMem1 = cellById.at(cellId1).cellFeature->volatileData;
	QByteArray cellMem2 = cellById.at(cellId2).cellFeature->volatileData;
	QByteArray cellMem3 = cellById.at(cellId3).cellFeature->volatileData;

	EXPECT_TRUE(bool(cellMem1.at(CommunicatorFunction::InternalDataSemantic::NewMessageReceived)));
	EXPECT_FALSE(bool(cellMem2.at(CommunicatorFunction::InternalDataSemantic::NewMessageReceived)));
	EXPECT_TRUE(bool(cellMem3.at(CommunicatorFunction::InternalDataSemantic::NewMessageReceived)));

	EXPECT_EQ(100, cellMem1.at(CommunicatorFunction::InternalDataSemantic::MessageCode));
	EXPECT_EQ(100, cellMem3.at(CommunicatorFunction::InternalDataSemantic::MessageCode));
	qreal receivedAngle1 = PhysicalQuantityConverter::convertDataToAngle(cellMem1.at(CommunicatorFunction::InternalDataSemantic::OriginAngle));
	qreal receivedAngle3 = PhysicalQuantityConverter::convertDataToAngle(cellMem3.at(CommunicatorFunction::InternalDataSemantic::OriginAngle));

	EXPECT_TRUE(std::abs(180 - receivedAngle1) < 2.0);
	EXPECT_TRUE(std::abs(-135  - receivedAngle3) < 2.0);
	/*
	{
		//create cells, cell functions and token for cluster1
		qreal cellEnergy = 0.0;
		int maxConnections = 0;
		int tokenAccessNumber = 0;

		QVector2D relPos = QVector2D();
		_cellWithToken = facade->buildFeaturedCell(cellEnergy, Enums::CellFunction::COMMUNICATOR, _context, maxConnections, tokenAccessNumber, relPos);
		_communicator1a = _cellWithToken->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

		relPos = QVector2D(0.0, 1.0, 0.0);
		_cellWithoutToken = facade->buildFeaturedCell(cellEnergy, Enums::CellFunction::COMMUNICATOR, _context, maxConnections, tokenAccessNumber, relPos);
		_communicator1b = _cellWithoutToken->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

		qreal tokenEnergy = 0.0;
		_token = facade->buildToken(_context, tokenEnergy);
		_cellWithToken->addToken(_token);

		//create cluster1
		QList< Cell* > cells;
		cells << _cellWithToken;
		cells << _cellWithoutToken;
		QVector2D pos(500.0, 500.0, 0.0);
		QVector2D vel(0.0, 0.0, 0.0);
		qreal angle = 0.0;
		qreal angularVel = 0.0;
		_cluster1 = facade->buildCellCluster(cells, angle, pos, angularVel, vel, _context);
	}

	{
		//create cell und cell function for cluster2
		qreal cellEnergy = 0.0;
		int maxConnections = 0;
		int tokenAccessNumber = 0;

		QVector2D relPos = QVector2D();
		Cell* cell = facade->buildFeaturedCell(cellEnergy, Enums::CellFunction::COMMUNICATOR, _context, maxConnections, tokenAccessNumber, relPos);
		_communicator2 = cell->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

		//create cluster2 within communication range
		QList< Cell* > cells;
		cells << cell;
		qreal distanceFromCluster1 = _context->getSimulationParameters()->cellFunctionCommunicatorRange / 2.0;
		QVector2D pos(500.0 + distanceFromCluster1, 500.0, 0.0);
		QVector2D vel(0.0, 0.0, 0.0);
		qreal angle = 0.0;
		qreal angularVel = 0.0;
		_cluster2 = facade->buildCellCluster(cells, angle, pos, angularVel, vel, _context);
	}

	//draw cells
	_cluster1->drawCellsToMap();
	_cluster2->drawCellsToMap();

	//init
	_communicator1a->getReceivedMessageRef() = CellFunctionCommunicatorImpl::MessageData();
	_communicator1a->getNewMessageReceivedRef() = false;
	_communicator1b->getReceivedMessageRef() = CellFunctionCommunicatorImpl::MessageData();
	_communicator1b->getNewMessageReceivedRef() = false;
	_communicator2->getReceivedMessageRef() = CellFunctionCommunicatorImpl::MessageData();
	_communicator2->getNewMessageReceivedRef() = false;
	for (int i = 0; i < 256; ++i)
		_token->getMemoryRef()[i] = 0;
*/
}

/*
TEST_F (TestCellFunctionCommunicator, testSendMessage)
{
	//setup channel
	quint8 channel = 1;
	quint8 differentChannel = 2;
	_communicator1b->getReceivedMessageRef().channel = channel;
	_communicator2->getReceivedMessageRef().channel = channel;

	//program token
	quint8 message = 100;
	quint8 angle = CodingPhysicalQuantities::convertAngleToData(180.0);
	quint8 distance = _context->getSimulationParameters()->cellFunctionCommunicatorRange / 2;
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN)] = static_cast<int>(Enums::CommunicatorIn::SEND_MESSAGE);
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN_CHANNEL)] = channel;
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN_MESSAGE)] = message;
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN_ANGLE)] = angle;
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN_DISTANCE)] = distance;

	//1. test: message received?
	_communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
	ASSERT_TRUE(_communicator2->getNewMessageReceivedRef()) << "No message received.";

	//2. test: correct angle received?
	qreal receivedAngle = CodingPhysicalQuantities::convertDataToAngle(_communicator2->getReceivedMessageRef().angle);
	QString s = QString("Message received with wrong angle; received angle: %1, expected angle: %2").arg(receivedAngle).arg(-45.0);
	ASSERT_TRUE(qAbs(receivedAngle - (-45.0)) < 2.0) << s.toLatin1().data();

	//3. test: correct angle received for an other direction?
	angle = CodingPhysicalQuantities::convertAngleToData(0.0);
	_token->getMemoryRef()[static_cast<int>(Enums::Communicator::IN_ANGLE)] = angle;
	_communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
	receivedAngle = CodingPhysicalQuantities::convertDataToAngle(_communicator2->getReceivedMessageRef().angle);
	s = QString("Message received with wrong angle; received angle: %1, expected angle: %2").arg(receivedAngle).arg(-135.0);
	ASSERT_TRUE(qAbs(receivedAngle - (-135.0)) < 2.0) << s.toLatin1().data();

	//4. test: two messages sent?
	quint8 numMsg = _token->getMemoryRef()[static_cast<int>(Enums::Communicator::OUT_SENT_NUM_MESSAGE)];
	s = QString("Wrong number messages sent. Messages sent: %1, should be 2.").arg(numMsg);
	ASSERT_EQ(numMsg, 2) << s.toLatin1().data();

	//5. test: one receiver has different channel => only one message sent?
	_communicator2->getReceivedMessageRef().channel = differentChannel;
	_communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
	numMsg = _token->getMemoryRef()[static_cast<int>(Enums::Communicator::OUT_SENT_NUM_MESSAGE)];
	s = QString("Wrong number messages sent. Messages sent: %1, should be 1.").arg(numMsg);
	ASSERT_EQ(numMsg, 1) << s.toLatin1().data();
}
*/

CommunicatorFunctionTest::~CommunicatorFunctionTest()
{
	delete _controller;
	delete _access;
}
