//#include "testaliencellcluster.h"

#include <QtTest/QtTest>
#include "aliencellcluster.h"
#include "../../globaldata/simulationsettings.h"

class TestAlienCellCluster : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase ();
    void testCreation ();
    void testCellVelocityDecomposition ();
    void cleanupTestCase ();

private:
    AlienGrid* _grid;
    AlienCellCluster* _cluster;

};


//implementation
void TestAlienCellCluster::initTestCase()
{
    _grid = new AlienGrid();
    _grid->init(1000, 1000);
}

void TestAlienCellCluster::testCreation()
{
    QList< AlienCell* > cells;
    for(int i = 0; i <= 100; ++i) {
        AlienCell* cell = new AlienCell(100.0, _grid);
        cell->setRelPos(QVector3D(i, 0.0, 0.0));
        cells << cell;
    }
    QVector3D pos(200.0, 100.0, 0.0);
    _cluster = new AlienCellCluster(cells, 0.0, pos, 0.0, QVector3D(), _grid);
    QCOMPARE(_cluster->getPosition().x(), 250.0);
    QCOMPARE(_cluster->getPosition().y(), 100.0);
}

void TestAlienCellCluster::testCellVelocityDecomposition()
{
    //calc cell velocities and then the cluster velocity
    //and comparison with previous values (there should be no change)
    _cluster->setAngularVel(2.0);
    _cluster->setVel(QVector3D(1.0, -0.5, 0.0));
    _cluster->updateCellVel(false);
    _cluster->updateVel_angularVel_via_cellVelocities();
    QVERIFY(qAbs(_cluster->getAngularVel() - 2.0) < ALIEN_PRECISION);
    QVERIFY(qAbs(_cluster->getVel().x() - 1.0) < ALIEN_PRECISION);
    QVERIFY(qAbs(_cluster->getVel().y() - (-0.5)) < ALIEN_PRECISION);
}

void TestAlienCellCluster::cleanupTestCase()
{
    delete _cluster;
    delete _grid;
}


/*
QTEST_MAIN(TestAlienCellCluster)
#include "testaliencellcluster.moc"
*/
