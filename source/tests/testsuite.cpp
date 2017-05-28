#include <gtest/gtest.h>
#include <QApplication>

#include "Model/modelservices.h"

/*
#include "unittests/testphysics.h"
#include "unittests/testcellcluster.h"
#include "unittests/testtoken.h"
#include "unittests/testcellfunctioncommunicator.h"
#include "integrationtests/integrationtestreplicator.h"
#include "integrationtests/integrationtestcomparison.h"
*/

//--gtest_filter=TestPhysics*
// Ex.: SquareRoot*

int main(int argc, char** argv) {
	ModelServices _modelServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
