#include <gtest/gtest.h>
#include <QApplication>

#include "ModelBasic/ModelBasicServices.h"
#include "ModelCpu/ModelCpuServices.h"

int main(int argc, char** argv) {
	ModelBasicServices _modelBasicServices;
	ModelCpuServices _modelCpuServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
