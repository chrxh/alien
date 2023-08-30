#include <iostream>

#include "Base/LoggingService.h"
#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationControllerImpl.h"
#include "Base/Resources.h"

#include "MainWindow.h"
#include "SimpleLogger.h"
#include "FileLogger.h"

namespace
{
    auto const generalInformation =
        "Please make sure that:\n\n1) You have an Nvidia graphics card with compute capability 6.0 or higher (for example "
        "GeForce 10 series).\n\n2) You have the latest Nvidia graphics driver installed.\n\n3) The name of the "
        "installation directory (including the parent directories) should not contain non-English characters. If this is not fulfilled, "
        "please re-install ALIEN to a suitable directory. Do not move the files manually.\n\n4) ALIEN needs write access to its own "
        "directory. This should normally be the case.\n\n5) If you have multiple graphics cards, please check that your primary monitor is "
        "connected to the CUDA-powered card. ALIEN uses the same graphics card for computation as well as rendering and chooses the one "
        "with the highest compute capability.\n\n6) If you possess both integrated and dedicated graphics cards, please ensure that the alien-executable is "
        "configured to use your high-performance graphics card. On Windows you need to access the 'Graphics settings,' add 'alien.exe' to the list, click "
        "'Options,' and choose 'High performance'.\n\nIf these conditions are not met, ALIEN may crash unexpectedly.";
}

int main(int, char**)
{
    SimpleLogger logger = std::make_shared<_SimpleLogger>();
    FileLogger fileLogger = std::make_shared<_FileLogger>();

    SimulationController simController;
    MainWindow mainWindow;

    try {
        simController = std::make_shared<_SimulationControllerImpl>();
        mainWindow = std::make_shared<_MainWindow>(simController, logger);
        mainWindow->mainLoop();
        mainWindow->shutdown();

    } catch (std::exception const& e) {
        auto message = std::string("The following exception occurred: ")
            + e.what();
        log(Priority::Important, message);
        std::cerr << message
                  << std::endl
                  << std::endl
                  << "See log.txt for more detailed information."
                  << std::endl
                  << std::endl
                  << generalInformation
                  << std::endl;
    } catch (...) {
        auto message = std::string("An unknown exception occurred.");
        log(Priority::Important, message);
        std::cerr << message << std::endl
                  << std::endl
                  << "See log.txt for more detailed information." << std::endl
                  << std::endl
                  << generalInformation << std::endl;
    }
    return 0;
}
