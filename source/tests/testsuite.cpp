#include <QtTest>
#include <QTextCodec>

#include "testphysics.h"
#include "testaliencellcluster.h"
#include "testalientoken.h"
#include "testaliencellfunctioncommunicator.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    TestPhysics testPhysics;
    TestAlienCellCluster testCellCluster;
    TestAlienToken testToken;
    TestAlienCellFunctionCommunicator testCommunicator;

    return QTest::qExec(&testPhysics, argc, argv)
            | QTest::qExec(&testCellCluster, argc, argv)
            | QTest::qExec(&testToken, argc, argv)
            | QTest::qExec(&testCommunicator, argc, argv);
}
