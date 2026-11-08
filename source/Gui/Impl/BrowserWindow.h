#pragma once

#include <optional>
#include <string>

#include <Network/NetworkResourceTreeTO.h>

#include <Persister/Interface/Definitions.h>
#include <Persister/Interface/DownloadCache.h>

#include "AlienWindow.h"
#include "BrowserData.h"
#include "Definitions.h"

// Shows the simulations of this computer and the simulations and genomes of the server: the workspaces on the left, their resources in the
// middle, the selected resource on the right and the simulators beside them
class BrowserWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(BrowserWindow);

public:
    void onRefresh();
    void onPreviewPictureChanged(std::string const& resourceId);
    void showResource(NetworkResourceType resourceType, WorkspaceType workspaceType, std::string const& resourceId);
    WorkspaceType getCurrentWorkspaceType() const;

    DownloadCache const& getSimulationCache() const;

private:
    BrowserWindow();

    void initIntern() override;
    void shutdownIntern() override;

    void processIntern() override;
    void processBackground() override;
    void processActivated() override;

    void processToolbar();
    void processWorkspace();
    void processWorkspaceSelection(float width, float height);
    void processWorkspaceItem(WorkspaceType workspaceType, std::string const& icon, std::string const& name, std::string const& tooltip);
    void processResourceView();
    void processHeader();
    void processFooter();
    bool isLoginRequired() const;
    float calcAccountChipWidth() const;
    void processAccountChip();
    void processUserList();
    void processStatusBar();

    void processEmojiWindow();
    void processEmojiButton(int emojiType);

    void processRefreshingScreen(RealVector2D const& startPos);

    void updateLocalWorkspaces();
    void selectWorkspace(WorkspaceType workspaceType);

    void onSave();
    void onEditResource(NetworkResourceTreeTO const& treeTO);
    void onChangeVisibility(NetworkResourceTreeTO const& treeTO);
    void onMoveResource(NetworkResourceTreeTO const& treeTO);
    void onDeleteResource(NetworkResourceTreeTO const& treeTO);
    void onShareSimulation(NetworkResourceRawTO const& rawTO);
    void onExpandFolders();
    void onCollapseFolders();
    void revealResource(std::string const& resourceId);

    BrowserData _data;
    BrowserPictureCache _pictureCache;
    BrowserGalleryWidget _galleryWidget;
    BrowserTableWidget _tableWidget;
    BrowserDetailsWidget _detailsWidget;
    BrowserUserListWidget _userListWidget;
    BrowserLoginHintWidget _loginHintWidget;
    BrowserLoginBannerWidget _loginBannerWidget;

    bool _galleryView = true;
    bool _showAllEmojis = false;
    float _userTableWidth = 0;
    std::optional<std::string> _resourceIdToShow;

    WorkspaceType _lastServerWorkspaceType = WorkspaceType_AlienProject;
    std::optional<int> _libraryRevision;
};
