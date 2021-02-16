#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/BaseServices.h"
#include "EngineInterface/SimulationAccess.h"
#include "EngineInterface/EngineInterfaceBuilderFacade.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SymbolTable.h"
#include "EngineInterface/EngineInterfaceServices.h"

#include "EngineGpu/EngineGpuServices.h"

#include "Web/WebServices.h"

#include "Gui/MainController.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("alien");
	QCoreApplication::setApplicationName("alien");

    BaseServices baseServices;
    EngineInterfaceServices engineInterfaceServices;
	EngineGpuServices engineGpuServices;
    WebServices webServices;

    MainController controller;
	controller.init();
	return a.exec();
}

