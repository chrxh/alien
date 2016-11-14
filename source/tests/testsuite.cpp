#include <QtTest>
#include <QTextCodec>

#include "testphysics.h"
#include "testcellcluster.h"
#include "testtoken.h"
#include "testcellfunctioncommunicator.h"
#include "integrationtestsimulation.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    TestPhysics testPhysics;
    TestCellCluster testCellCluster;
    TestToken testToken;
    TestCellFunctionCommunicator testCommunicator;

    IntegrationTestSimulation intTestSimulation;

    return QTest::qExec(&testPhysics, argc, argv)
            | QTest::qExec(&testCellCluster, argc, argv)
            | QTest::qExec(&testToken, argc, argv)
            | QTest::qExec(&testCommunicator, argc, argv)
            | QTest::qExec(&intTestSimulation, argc, argv);
}
