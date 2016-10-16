#include <QtTest>
#include <QTextCodec>

#include "testphysics.h"
#include "testaliencellcluster.h"
#include "testalientoken.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    TestPhysics testPhysics;
    TestAlienCellCluster testAlienCellCluster;
    TestAlienToken testAlienToken;

    return QTest::qExec(&testPhysics, argc, argv)
            | QTest::qExec(&testAlienCellCluster, argc, argv)
            | QTest::qExec(&testAlienToken, argc, argv);
}
