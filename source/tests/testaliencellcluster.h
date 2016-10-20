#ifndef TESTALIENCELLCLUSTER_H
#define TESTALIENCELLCLUSTER_H

#include <QtTest/QtTest>
#include "simulation/entities/aliencellcluster.h"
#include "testsettings.h"

class TestAlienCellCluster : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase()
    {
        _grid = new AlienGrid();
        _grid->init(1000, 1000);
    }

    void testCreation()
    {
        QList< AlienCell* > cells;
        for(int i = 0; i <= 100; ++i) {
            AlienCell* cell = AlienCell::buildCell(100.0, _grid);
            cell->setRelPos(QVector3D(i, 0.0, 0.0));
            cells << cell;
        }
        QVector3D pos(200.0, 100.0, 0.0);
        _cluster = AlienCellCluster::buildCellCluster(cells, 0.0, pos, 0.0, QVector3D(), _grid);
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

    void cleanupTestCase()
    {
        delete _cluster;
        delete _grid;
    }

private:
    AlienGrid* _grid;
    AlienCellCluster* _cluster;

};



#endif
