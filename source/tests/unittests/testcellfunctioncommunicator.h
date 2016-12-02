#ifndef TESTCELLFUNCTIONCOMMUNICATOR_H
#define TESTCELLFUNCTIONCOMMUNICATOR_H

#include "testsettings.h"

#include "model/entities/grid.h"
#include "model/entities/cellcluster.h"
#include "model/entities/cell.h"
#include "model/entities/token.h"
#include "model/features/_impl/cellfunctioncommunicatorimpl.h"
#include "model/factoryfacade.h"
#include "global/servicelocator.h"

#include <QtTest/QtTest>

class TestCellFunctionCommunicator : public QObject
{
    Q_OBJECT
private slots:
    void initTestCase()
    {
        _grid = new Grid();
        _grid->init(1000, 1000);
        FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();

        {
            //create cells, cell functions and token for cluster1
            qreal cellEnergy = 0.0;
            int maxConnections = 0;
            int tokenAccessNumber = 0;

            QVector3D relPos = QVector3D();
            _cellWithToken = facade->buildFeaturedCell(cellEnergy, CellFunctionType::COMMUNICATOR, _grid, maxConnections, tokenAccessNumber, relPos);
            _communicator1a = _cellWithToken->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

            relPos = QVector3D(0.0, 1.0, 0.0);
            _cellWithoutToken = facade->buildFeaturedCell(cellEnergy, CellFunctionType::COMMUNICATOR, _grid, maxConnections, tokenAccessNumber, relPos);
            _communicator1b = _cellWithoutToken->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

            qreal tokenEnergy = 0.0;
            _token = new Token(tokenEnergy);
            _cellWithToken->addToken(_token);

            //create cluster1
            QList< Cell* > cells;
            cells << _cellWithToken;
            cells << _cellWithoutToken;
            QVector3D pos(500.0, 500.0, 0.0);
            QVector3D vel(0.0, 0.0, 0.0);
            qreal angle = 0.0;
            qreal angularVel = 0.0;
            _cluster1 = facade->buildCellCluster(cells, angle, pos, angularVel, vel, _grid);
        }

        {
            //create cell und cell function for cluster2
            qreal cellEnergy = 0.0;
            int maxConnections = 0;
            int tokenAccessNumber = 0;

            QVector3D relPos = QVector3D();
            Cell* cell = facade->buildFeaturedCell(cellEnergy, CellFunctionType::COMMUNICATOR, _grid, maxConnections, tokenAccessNumber, relPos);
            _communicator2 = cell->getFeatures()->findObject<CellFunctionCommunicatorImpl>();

            //create cluster2 within communication range
            QList< Cell* > cells;
            cells << cell;
            qreal distanceFromCluster1 = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE/2.0;
            QVector3D pos(500.0+distanceFromCluster1, 500.0, 0.0);
            QVector3D vel(0.0, 0.0, 0.0);
            qreal angle = 0.0;
            qreal angularVel = 0.0;
            _cluster2 = facade->buildCellCluster(cells, angle, pos, angularVel, vel, _grid);
        }

        //draw cells
        _cluster1->drawCellsToMap();
        _cluster2->drawCellsToMap();
    }

    void init ()
    {
        _communicator1a->_receivedMessage.channel = 0;
        _communicator1a->_receivedMessage.message = 0;
        _communicator1a->_receivedMessage.angle = 0;
        _communicator1a->_receivedMessage.distance = 0;
        _communicator1a->_newMessageReceived = false;
        _communicator1b->_receivedMessage.channel = 0;
        _communicator1b->_receivedMessage.message = 0;
        _communicator1b->_receivedMessage.angle = 0;
        _communicator1b->_receivedMessage.distance = 0;
        _communicator1b->_newMessageReceived = false;
        _communicator2->_receivedMessage.channel = 0;
        _communicator2->_receivedMessage.message = 0;
        _communicator2->_receivedMessage.angle = 0;
        _communicator2->_receivedMessage.distance = 0;
        _communicator2->_newMessageReceived = false;
        for(int i = 0; i < 256; ++i)
            _token->memory[i] = 0;
    }

    void testSendMessage ()
    {
        //setup channel
        quint8 channel = 1;
        quint8 differentChannel = 2;
        _communicator1b->_receivedMessage.channel = channel;
        _communicator2->_receivedMessage.channel = channel;

        //program token
        quint8 message = 100;
        quint8 angle = CellFunction::convertAngleToData(180.0);
        quint8 distance = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE/2;
        _token->memory[static_cast<int>(COMMUNICATOR::IN)] = static_cast<int>(COMMUNICATOR_IN::SEND_MESSAGE);
        _token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)] = channel;
        _token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)] = message;
        _token->memory[static_cast<int>(COMMUNICATOR::IN_ANGLE)] = angle;
        _token->memory[static_cast<int>(COMMUNICATOR::IN_DISTANCE)] = distance;

        //1. test: message received?
        _communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
        QVERIFY2(_communicator2->_newMessageReceived, "No message received.");

        //2. test: correct angle received?
        qreal receivedAngle = CellFunction::convertDataToAngle(_communicator2->_receivedMessage.angle);
        QString s = QString("Message received with wrong angle; received angle: %1, expected angle: %2").arg(receivedAngle).arg(-45.0);
        QVERIFY2(qAbs(receivedAngle - (-45.0)) < 2.0, s.toLatin1().data());

        //3. test: correct angle received for an other direction?
        angle = CellFunction::convertAngleToData(0.0);
        _token->memory[static_cast<int>(COMMUNICATOR::IN_ANGLE)] = angle;
        _communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
        receivedAngle = CellFunction::convertDataToAngle(_communicator2->_receivedMessage.angle);
        s = QString("Message received with wrong angle; received angle: %1, expected angle: %2").arg(receivedAngle).arg(-135.0);
        QVERIFY2(qAbs(receivedAngle - (-135.0)) < 2.0, s.toLatin1().data());

        //4. test: two messages sent?
        quint8 numMsg = _token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)];
        s = QString("Wrong number messages sent. Messages sent: %1, should be 2.").arg(numMsg);
        QVERIFY2(numMsg == 2, s.toLatin1().data());

        //5. test: one receiver has different channel => only one message sent?
        _communicator2->_receivedMessage.channel = differentChannel;
        _communicator1a->process(_token, _cellWithToken, _cellWithoutToken);
        numMsg = _token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)];
        s = QString("Wrong number messages sent. Messages sent: %1, should be 1.").arg(numMsg);
        QVERIFY2(numMsg == 1, s.toLatin1().data());
    }

    void cleanupTestCase()
    {
        delete _cluster1;
        delete _cluster2;
        delete _grid;
    }

private:
    Grid* _grid;

    //data for cluster1
    CellCluster* _cluster1;
    Cell* _cellWithToken;
    Cell* _cellWithoutToken;
    CellFunctionCommunicatorImpl* _communicator1a;
	CellFunctionCommunicatorImpl* _communicator1b;
    Token* _token;

    //data for cluster2
    CellCluster* _cluster2;
	CellFunctionCommunicatorImpl* _communicator2;
};

#endif // TESTCELLFUNCTIONCOMMUNICATOR_H
