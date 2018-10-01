#include <gtest/gtest.h>
#include <QApplication>

#include "ModelCpu/ModelCpuServices.h"

int main(int argc, char** argv) {
	ModelCpuServices _modelServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
