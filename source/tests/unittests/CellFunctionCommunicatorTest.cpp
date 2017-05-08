#include <gtest/gtest.h>

#include "global/ServiceLocator.h"
#include "model/entities/CellCluster.h"
#include "model/entities/Cell.h"
#include "model/entities/Token.h"
#include "model/features/_impl/CellFunctionCommunicatorImpl.h"
#include "model/physics/CodingPhysicalQuantities.h"
#include "model/context/UnitContext.h"
#include "model/BuilderFacade.h"
#include "model/ModelSettings.h"
#include "model/context/SpaceMetric.h"
#include "model/context/SimulationParameters.h"

#include "tests/TestSettings.h"

class CellFunctionCommunicatorTest : public ::testing::Test
{
public:
	CellFunctionCommunicatorTest();
	~CellFunctionCommunicatorTest();

protected:
    UnitContext* _context;

    //data for cluster1
    CellCluster* _cluster1 = nullptr;
    Cell* _cellWithToken = nullptr;
    Cell* _cellWithoutToken = nullptr;
    CellFunctionCommunicatorImpl* _communicator1a = nullptr;
	CellFunctionCommunicatorImpl* _communicator1b = nullptr;
    Token* _token = nullptr;

    //data for cluster2
    CellCluster* _cluster2 = nullptr;
	CellFunctionCommunicatorImpl* _communicator2 = nullptr;
};

CellFunctionCommunicatorTest::CellFunctionCommunicatorTest()
{
/*
	BuilderFacade* facade = ServiceLocator::getInstance().getService<BuilderFacade>();

	_context = facade->buildSimulationContext();
	auto metric = facade->buildSpaceMetric();
	metric->init({ 1000, 1000 });
	_context->init(metric);

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

CellFunctionCommunicatorTest::~CellFunctionCommunicatorTest()
{
	delete _cluster1;
	delete _cluster2;
	delete _context;
}
