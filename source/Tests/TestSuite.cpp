#include <gtest/gtest.h>
#include <QApplication>

#include "Base/BaseServices.h"
#include "EngineInterface/EngineInterfaceServices.h"
#include "EngineGpu/EngineGpuServices.h"

int main(int argc, char** argv) {
    BaseServices baseServices;
    EngineInterfaceServices _EngineInterfaceServices;
	EngineGpuServices _EngineGpuServices;

    QApplication app(argc, argv);

	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
