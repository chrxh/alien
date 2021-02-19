#include <sstream>
#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>
#include <QMessageBox>

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

#include "MainController.h"
#include "Settings.h"
#include "FileLogger.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("alien");
	QCoreApplication::setApplicationName("alien");

    BaseServices baseServices;
    EngineInterfaceServices engineInterfaceServices;
	EngineGpuServices engineGpuServices;
    WebServices webServices;

    FileLogger fileLogger;

    MainController controller;
	controller.init();
    try {
	    return a.exec();
    }
    catch(std::exception const& e) {

        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, e.what());

        QMessageBox messageBox;
        messageBox.critical(0, "Critical error", QString::fromStdString(e.what()));
        exit(EXIT_FAILURE);
    }
}

