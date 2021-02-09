#include <gtest/gtest.h>
#include <QApplication>

#include "EngineInterface/ModelBasicServices.h"
#include "EngineGpu/ModelGpuServices.h"

int main(int argc, char** argv) {
	ModelBasicServices _modelBasicServices;
	ModelGpuServices _modelGpuServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
