#pragma once

#include <string>

#include <Base/Interface/Resources.h>

namespace Const
{
    inline std::string const& getGeneralInformation()
    {
        static std::string const result =
            "Please make sure that:\n\n1) You have an NVIDIA graphics card with compute capability 7.5 or higher (for example "
            "GeForce RTX 20 series).\n\n2) You have the latest NVIDIA graphics driver installed.\n\n3) The name of the "
            "installation directory (including the parent directories) should not contain non-English characters. If this is not fulfilled, "
            "please re-install ALIEN to a suitable directory. Do not move the files manually. If you use Windows, make also sure that you install ALIEN with a "
            "Windows user that contains no non-English characters. If this is not the case, a new Windows user could be created to solve this "
            "problem.\n\n4) ALIEN "
            "needs write access to its own "
            "directory. This should normally be the case.\n\n5) If you have multiple graphics cards, please check that your primary monitor is "
            "connected to the CUDA-powered card. ALIEN uses the same graphics card for computation as well as rendering and chooses the one "
            "with the highest compute capability.\n\n6) If you possess both integrated and dedicated graphics cards, please ensure that the "
            "alien-executable is "
            "configured to use your high-performance graphics card. On Windows you need to access the 'Graphics settings,' add 'alien.exe' to the list, click "
            "'Options,' and choose 'High performance'.\n\nIf these conditions are not met, ALIEN may crash unexpectedly.\n\n"
            "If the conditions are met and the error still occurs, please enable Settings -> Debug mode in the menu bar, try to reproduce the error and then "
            "create a GitHub issue on https://github.com/chrxh/alien/issues where "
            + Const::LogFilename.string() + " and " + Const::TraceFilename.string() + " are attached.\n\n";
        return result;
    }
}
