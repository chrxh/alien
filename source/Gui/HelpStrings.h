#pragma once

#include <string>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Resources.h>

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

    std::string const NotAllowedCharacters = "Your input contains not allowed characters.";

    std::string const CellEnergyTooltip = "The amount of internal energy of the cell. The cell undergoes decay when its energy falls below a critical "
                                          "threshold (refer to the 'Minimum energy' simulation parameter).";

    std::string const CreatorPencilRadiusTooltip = "The radius of the pencil in number of solid objects.";

    std::string const CreatorDrawingTypeTooltip =
        "Specifies whether the drawn solid objects should form a solid body (with connections) or a fluid (without connections).";

    std::string const CreatorRectangleWidthTooltip = "The width of the rectangle in cells.";

    std::string const CreatorRectangleHeightTooltip = "The height of the rectangle in cells.";

    std::string const CreatorHexagonLayersTooltip = "The number of layers in cells starting from the center.";

    std::string const CreatorDiscOuterRadiusTooltip = "The outer radius of the disc in cells.";

    std::string const CreatorDiscInnerRadiusTooltip = "The inner radius of the disc in cells.";

    std::string const CreatorDistanceTooltip = "The distance between two connected cells.";

    std::string const LoginHowToCreateNewUseTooltip = "Please enter the desired user name and password and proceed by clicking the 'Create user' button.";

    std::string const LoginForgotYourPasswordTooltip = "Please enter the user name and proceed by clicking the 'Reset password' button.";

    std::string const LoginSecurityInformationTooltip =
        "The data transfer to the server is encrypted via https. On the server side, the password is not stored in cleartext, but as a salted SHA-256 hash "
        "value in the database. If the toggle 'Remember' is activated, the password will be stored in the Windows registry under the path "
        "'HKEY_CURRENT_USER\\SOFTWARE\\alien' "
        "or, in the case of other OS, in 'settings.json' on your local machine.";

    std::string const LoginRememberTooltip = "If the toggle 'Remember' is activated, the password will be stored in the Windows registry under the path "
                                             "'HKEY_CURRENT_USER\\SOFTWARE\\alien' or, in the case of other OS, in 'settings.json' on your local machine. It "
                                             "is recommended not to choose a password that is used elsewhere.";

    std::string const LoginShareGpuInfoTooltip1 =
        "If this option is enabled, other users will be able to see in the browser window that you have the following graphics card: ";
    std::string const LoginShareGpuInfoTooltip2 = "As a result, you will be able to see the GPU information of other registered users who have shared it.";

    std::string const BrowserFeaturedWorkspaceTooltip =
        "This workspace is curated by the alien-project and contains the simulations that come along with the released versions. They cover a wide range and "
        "exploit different features.";

    std::string const BrowserCommunityWorkspaceTooltip =
        "All logged-in users can share their simulations and genomes here. The files stored in this workspace are visible to all users.";

    std::string const BrowserPrivateWorkspaceTooltip =
        "Each user account has its own private space. The simulations and genomes are only visible to the logged-in user.";

    std::string const BrowserLoginChipTooltip = "Log in or create a new account to upload your own simulations and genomes and to react to those of others.";
}
