#ifndef TESTALIENCELLFUNCTIONCOMMUNICATOR_H
#define TESTALIENCELLFUNCTIONCOMMUNICATOR_H

#include "testsettings.h"
#include "simulation/processing/aliencellfunctioncommunicator.h"
#include "simulation/entities/aliengrid.h"
#include "simulation/entities/aliencellcluster.h"
#include "simulation/entities/aliencell.h"
#include "simulation/entities/alientoken.h"

#include <QtTest/QtTest>

class TestAlienCellFunctionCommunicator : public QObject
{
    Q_OBJECT
private slots:
    void initTestCase()
    {
        _grid = new AlienGrid();
        _grid->init(1000, 1000);

        {
            //create cells, cell functions and token for cluster1
            qreal cellEnergy = 0.0;
            int maxConnections = 0;
            int tokenAccessNumber = 0;
            _communicator1 = new AlienCellFunctionCommunicator(_grid);
            QVector3D relPos = QVector3D();
            _cellWithToken = AlienCell::buildCell(cellEnergy, _grid, maxConnections, tokenAccessNumber, _communicator1, relPos);
            AlienCellFunctionCommunicator* someOtherCellFunction = new AlienCellFunctionCommunicator(_grid);
            relPos = QVector3D(0.0, 1.0, 0.0);
            _cellWithoutToken = AlienCell::buildCell(cellEnergy, _grid, maxConnections, tokenAccessNumber, someOtherCellFunction, relPos);
            qreal tokenEnergy = 0.0;
            _token = new AlienToken(tokenEnergy);
            _cellWithToken->addToken(_token);

            //create cluster1
            QList< AlienCell* > cells;
            cells << _cellWithToken;
            cells << _cellWithoutToken;
            QVector3D pos(500.0, 500.0, 0.0);
            QVector3D vel(0.0, 0.0, 0.0);
            qreal angle = 0.0;
            qreal angularVel = 0.0;
            _cluster1 = AlienCellCluster::buildCellCluster(cells, angle, pos, angularVel, vel, _grid);
        }

        {
            //create cell und cell function for cluster2
            qreal cellEnergy = 0.0;
            int maxConnections = 0;
            int tokenAccessNumber = 0;
            _communicator2 = new AlienCellFunctionCommunicator(_grid);
            QVector3D relPos = QVector3D();
            AlienCell* cell = AlienCell::buildCell(cellEnergy, _grid, maxConnections, tokenAccessNumber, _communicator2, relPos);

            //create cluster2 within communication range
            QList< AlienCell* > cells;
            cells << cell;
            qreal distanceFromCluster1 = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE/2.0;
            QVector3D pos(500.0+distanceFromCluster1, 500.0, 0.0);
            QVector3D vel(0.0, 0.0, 0.0);
            qreal angle = 0.0;
            qreal angularVel = 0.0;
            _cluster2 = AlienCellCluster::buildCellCluster(cells, angle, pos, angularVel, vel, _grid);
        }

        //draw cells
        _cluster1->drawCellsToMap();
        _cluster2->drawCellsToMap();
    }

    void init ()
    {
        _communicator1->_receivedMessage.channel = 0;
        _communicator1->_receivedMessage.message = 0;
        _communicator1->_receivedMessage.angle = 0;
        _communicator1->_receivedMessage.distance = 0;
        _communicator1->_newMessageReceived = false;
        _communicator2->_receivedMessage.channel = 0;
        _communicator2->_receivedMessage.message = 0;
        _communicator2->_receivedMessage.angle = 0;
        _communicator2->_receivedMessage.distance = 0;
        _communicator2->_newMessageReceived = false;
        for(int i = 0; i < 256; ++i)
            _token->memory[i] = 0;
    }

    void testSendMessageToSameChannel ()
    {
        //setup channel
        quint8 channel = 1;
        _communicator2->_receivedMessage.channel = channel;

        //program token
        quint8 message = 100;
        quint8 angle = 0;
        quint8 distance = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE/2;
        _token->memory[static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN)] = static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR_IN::SEND_MESSAGE);
        _token->memory[static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_CHANNEL)] = channel;
        _token->memory[static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_MESSAGE)] = message;
        _token->memory[static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_ANGLE)] = angle;
        _token->memory[static_cast<int>(AlienCellFunctionCommunicator::COMMUNICATOR::IN_DISTANCE)] = distance;

        //send message
        AlienEnergy* energy = 0;
        bool decompose = false;
        _communicator1->execute(_token, _cellWithToken, _cellWithoutToken, energy, decompose);

        //message received?
        QVERIFY2(_communicator2->_newMessageReceived, "No message received.");

        //correct angle received?
        qreal receivedAngle = AlienCellFunction::convertDataToAngle(_communicator2->_receivedMessage.angle);
        QString s = QString("Message from wrong angle received; received angle: %1, expected angle: %2").arg(receivedAngle).arg(-45.0);
        QVERIFY2(qAbs(receivedAngle - (-45.0)) < 2.0, s.toLatin1().data());

        //further evaluation, e.g. numMsg...
    }

    void cleanupTestCase()
    {
        delete _cluster1;
        delete _cluster2;
        delete _grid;
    }

private:
    AlienGrid* _grid;

    //data for cluster1
    AlienCellCluster* _cluster1;
    AlienCell* _cellWithToken;
    AlienCell* _cellWithoutToken;
    AlienCellFunctionCommunicator* _communicator1;
    AlienToken* _token;

    //data for cluster2
    AlienCellCluster* _cluster2;
    AlienCellFunctionCommunicator* _communicator2;
};

#endif // TESTALIENCELLFUNCTIONCOMMUNICATOR_H
