//#include "testaliencellcluster.h"

#include <QtTest/QtTest>
#include "aliencellcluster.h"
#include "../../globaldata/simulationparameters.h"

class TestAlienCellCluster : public QObject
{
    Q_OBJECT
private slots:

    void initTestCase ();
    void testCreation ();
    void testCellVelocityDecomposition ();
    void cleanupTestCase ();

private:
    AlienGrid* grid;
    AlienCellCluster* cluster;

};


//implementation
void TestAlienCellCluster::initTestCase()
{
    grid = new AlienGrid();
    grid->init(1000, 1000);
}

void TestAlienCellCluster::testCreation()
{
    QList< AlienCell* > cells;
    for(int i = 0; i <= 100; ++i) {
        AlienCell* cell = new AlienCell(100.0);
        cell->setRelPos(QVector3D(i, 0.0, 0.0));
        cells << cell;
    }
    QVector3D pos(200.0, 100.0, 0.0);
    cluster = new AlienCellCluster(grid, cells, 0.0, pos, 0.0, QVector3D(0.0, 0.0, 0.0));
    QCOMPARE(cluster->getPosition().x(), 250.0);
    QCOMPARE(cluster->getPosition().y(), 100.0);
}

void TestAlienCellCluster::testCellVelocityDecomposition()
{
    //calc cell velocities and then the cluster velocity
    //and comparison with previous values (there should be no change)
    cluster->setAngularVel(2.0);
    cluster->setVel(QVector3D(1.0, -0.5, 0.0));
    cluster->updateCellVel(false);
    cluster->updateVel_angularVel_via_cellVelocities();
    QCOMPARE(qAbs(cluster->getAngularVel() - 2.0) < ALIEN_PRECISION, true);
    QCOMPARE(qAbs(cluster->getVel().x() - 1.0) < ALIEN_PRECISION, true);
    QCOMPARE(qAbs(cluster->getVel().y() - (-0.5)) < ALIEN_PRECISION, true);
}

void TestAlienCellCluster::cleanupTestCase()
{
    delete cluster;
    delete grid;
}


/*
QTEST_MAIN(TestAlienCellCluster)
#include "testaliencellcluster.moc"
*/
