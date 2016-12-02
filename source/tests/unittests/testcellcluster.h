#ifndef TESTCELLCLUSTER_H
#define TESTCELLCLUSTER_H

#include "testsettings.h"

#include "model/entities/grid.h"
#include "model/entities/token.h"
#include "model/factoryfacade.h"
#include "model/entities/cellcluster.h"
#include "global/servicelocator.h"

#include <QtTest/QtTest>

class TestCellCluster : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase()
    {
        _grid = new Grid();
        _grid->init(1000, 1000);
    }

    void testCreation()
    {
        QList< Cell* > cells;
        FactoryFacade* facade = ServiceLocator::getInstance().getService<FactoryFacade>();
        for(int i = 0; i <= 100; ++i) {
            Cell* cell = facade->buildFeaturedCell(100.0, CellFunctionType::COMPUTER, _grid);
            cell->setRelPos(QVector3D(i, 0.0, 0.0));
            cells << cell;
        }
        QVector3D pos(200.0, 100.0, 0.0);
        _cluster = facade->buildCellCluster(cells, 0.0, pos, 0.0, QVector3D(), _grid);
        QCOMPARE(_cluster->getPosition().x(), 250.0);
        QCOMPARE(_cluster->getPosition().y(), 100.0);
    }

    void testCellVelocityDecomposition()
    {
        //calc cell velocities and then the cluster velocity
        //and comparison with previous values (there should be no change)
        _cluster->setAngularVel(2.0);
        _cluster->setVel(QVector3D(1.0, -0.5, 0.0));
        _cluster->updateCellVel(false);
        _cluster->updateVel_angularVel_via_cellVelocities();
        QVERIFY(qAbs(_cluster->getAngularVel() - 2.0) < TEST_REAL_PRECISION);
        QVERIFY(qAbs(_cluster->getVel().x() - 1.0) < TEST_REAL_PRECISION);
        QVERIFY(qAbs(_cluster->getVel().y() - (-0.5)) < TEST_REAL_PRECISION);
    }

    void testNewConnections()
    {
        //take three arbitrary cells
        QVERIFY(_cluster->getCellsRef().size() > 3);
        _cell1 = _cluster->getCellsRef().at(0);
        _cell2 = _cluster->getCellsRef().at(1);
        _cell3 = _cluster->getCellsRef().at(2);
        _cell4 = _cluster->getCellsRef().at(3);
        _cell1->resetConnections(3);
        _cell2->resetConnections(1);
        _cell3->resetConnections(1);
        _cell4->resetConnections(1);

        //connect cells
        _cell1->newConnection(_cell2);
        _cell1->newConnection(_cell3);
        _cell1->newConnection(_cell4);
        QVERIFY(_cell1->getConnection(0) == _cell2);
        QVERIFY(_cell1->getConnection(1) == _cell3);
        QVERIFY(_cell1->getConnection(2) == _cell4);
        QVERIFY(_cell2->getConnection(0) == _cell1);
        QVERIFY(_cell3->getConnection(0) == _cell1);
        QVERIFY(_cell4->getConnection(0) == _cell1);
    }

    void testTokenSpreading()
    {
        _cell1->setTokenAccessNumber(0);
        _cell2->setTokenAccessNumber(1);
        _cell3->setTokenAccessNumber(1);
        _cell4->setTokenAccessNumber(0);
        Token* token = new Token(simulationParameters.MIN_TOKEN_ENERGY*3);
        _cell1->addToken(token, Cell::ACTIVATE_TOKEN::NOW, Cell::UPDATE_TOKEN_ACCESS_NUMBER::YES);
        QList< EnergyParticle* > tempEP;
        bool tempDecomp = false;
        _cluster->movementProcessingStep4(tempEP, tempDecomp);
        QVERIFY(_cell1->getNumToken(true) == 0);
        QVERIFY(_cell2->getNumToken(true) == 1);
        QVERIFY(_cell3->getNumToken(true) == 1);
        QVERIFY(_cell4->getNumToken(true) == 0);
    }

    void cleanupTestCase()
    {
        delete _cluster;
        delete _grid;
    }

private:
    Grid* _grid;
    CellCluster* _cluster;
    Cell* _cell1;
    Cell* _cell2;
    Cell* _cell3;
    Cell* _cell4;
};



#endif
