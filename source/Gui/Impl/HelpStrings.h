#pragma once

#include <string>

#include <Fonts/IconsFontAwesome5.h>

namespace Const
{
    std::string const NotAllowedCharacters = "Your input contains not allowed characters.";

    std::string const CreatorEnergyTooltip = "Energy of each created object or energy particle.";

    std::string const CreatorPencilRadiusTooltip = "The radius of the pencil in number of objects.";

    std::string const CreatorMaterialTooltip = "Material of the objects to be created.\n" ICON_FA_CHEVRON_RIGHT
                                               " Solid: inorganic rigid particles that are connected to form a solid body.\n" ICON_FA_CHEVRON_RIGHT
                                               " Fluid: inorganic freely flowing particles without connections.\n" ICON_FA_CHEVRON_RIGHT
                                               " Free cells: organic substance without a genome. It can serve as food.\n" ICON_FA_CHEVRON_RIGHT
                                               " Energy particles: energy that can be absorbed by cells.";

    std::string const CreatorRectangleWidthTooltip = "The number of objects in the horizontal direction.";

    std::string const CreatorRectangleHeightTooltip = "The number of objects in the vertical direction.";

    std::string const CreatorHexagonLayersTooltip = "The number of object layers, counted from the center.";

    std::string const CreatorDiscOuterRadiusTooltip = "The outer radius of the disc.";

    std::string const CreatorDiscInnerRadiusTooltip = "The inner radius of the disc. Objects are only created between the inner and the outer radius.";

    std::string const CreatorDistanceTooltip = "The distance between two neighboring objects.";

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

    std::string const BrowserSimulatorCountsTooltip =
        "Active: ALIEN instances running in the last 25 minutes.\n"
        "Last 24h: different IP addresses that ran ALIEN in the last 24 hours.\n\n"
        "Both include the command-line interface and instances without login. The server keeps only a hash of each IP address in memory and deletes it "
        "24 hours after the last report. Instances sharing one IP address count once. Disable 'Count me as active' in the network settings to be left out.";

    std::string const NetworkCountAsActiveTooltip =
        "If this option is enabled, this program and the command-line interface are included in the numbers of active simulators shown in the browser "
        "window. The server tells the instances apart by a hash of their IP address that it keeps only in memory for 24 hours and never stores.";

    std::string const BrowserLocalWorkspaceTooltip =
        "The simulations in one folder of this computer. Save (Ctrl+S) stores the current simulation there. Nobody else can see them until you share "
        "them.";

    std::string const BrowserLocalGenomesWorkspaceTooltip =
        "The genomes in one folder of this computer. Save genome stores the genome of the genome editor there. Nobody else can see them until you "
        "share them.";

    std::string const BrowserAutosaveWorkspaceTooltip =
        "The current simulation is saved here automatically. With limited save files, the oldest autosaves of the current subfolder are deleted.";

    std::string const BrowserFeaturedWorkspaceTooltip =
        "This workspace is curated by the alien-project and contains the simulations that come along with the released versions. They cover a wide range and "
        "exploit different features.";

    std::string const BrowserCommunityWorkspaceTooltip =
        "All logged-in users can share their simulations and genomes here. The files stored in this workspace are visible to all users.";

    std::string const BrowserPrivateWorkspaceTooltip = "Everything you have uploaded. Uploads that are not shared with the community are only visible to you.";

    std::string const BrowserLoginChipTooltip = "Log in or create a new account to upload your own simulations and genomes.";

    std::string const BrowserLikedWorkspaceTooltip =
        "Everything on the server that you gave a heart or another reaction, whether it is featured, shared by the community or your own.";

    std::string const BrowserGpuHiddenLoggedOutTooltip = "The GPU models are only visible to logged-in users who share their own GPU model.";
    std::string const BrowserGpuHiddenNotSharedTooltip = "The GPU models are only visible if you share your own GPU model.";
    std::string const BrowserShareGpuTooltip = "Other users will be able to see in the browser window that you have the following graphics card: ";
}
