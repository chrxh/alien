#include <gtest/gtest.h>
#include <QApplication>

#include "ModelBasic/ModelBasicServices.h"
#include "ModelCpu/ModelCpuServices.h"
#include "ModelGpu/ModelGpuServices.h"

int main(int argc, char** argv) {
	ModelBasicServices _modelBasicServices;
	ModelCpuServices _modelCpuServices;
	ModelGpuServices _modelGpuServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
