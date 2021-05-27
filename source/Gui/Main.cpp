#include <sstream>
#include <QApplication>
#include <QtCore/qmath.h>
#include <QVector2D>
#include <QMessageBox>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "Base/BaseServices.h"
#include "Base/Exceptions.h"
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
#include "BugReportController.h"
#include "BugReportLogger.h"

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
	QCoreApplication::setOrganizationName("alien");
	QCoreApplication::setApplicationName("alien");

    BaseServices baseServices;
    EngineInterfaceServices engineInterfaceServices;
	EngineGpuServices engineGpuServices;
    WebServices webServices;

    FileLogger fileLogger;
    BugReportLogger bugReportLogger;

    MainController controller;

    try {
        controller.init();
        return a.exec();
    } catch (SystemRequirementNotMetException const& e) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, e.what());

        std::stringstream ss;
        ss << "Your system does not meet the minimum system requirements." << std::endl
           << "Error message: " << e.what();

        QMessageBox::critical(nullptr, "System requirements", QString::fromStdString(ss.str()));

        exit(EXIT_FAILURE);
    } catch (SpecificCudaException const& e) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, e.what());

        QMessageBox::critical(nullptr, "Error", QString::fromStdString(e.what()));

        exit(EXIT_FAILURE);
    } catch (BugReportException const& e) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, e.what());

        BugReportController bugReportController(e.what(), bugReportLogger.getFullProtocol());
        bugReportController.execute();

        exit(EXIT_FAILURE);
    } catch(std::exception const& e) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, e.what());

        BugReportController bugReportController(e.what(), bugReportLogger.getFullProtocol());
        bugReportController.execute();

        exit(EXIT_FAILURE);
    } catch (...) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        auto message = "Unknown exception thrown.";
        loggingService->logMessage(Priority::Important, message);

        BugReportController bugReportController(message, bugReportLogger.getFullProtocol());
        bugReportController.execute();
        exit(EXIT_FAILURE);
    }
}

