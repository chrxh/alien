#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/SymbolTable.h"
#include "ModelBasic/ModelBasicServices.h"

#include "ModelGpu/ModelGpuServices.h"

#include "Gui/MainController.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("alien");
	QCoreApplication::setApplicationName("alien");

	ModelBasicServices modelBasicServices;
	ModelGpuServices modelGpuServices;

    MainController controller;
	controller.init();
	return a.exec();
}

