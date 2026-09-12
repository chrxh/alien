#pragma once

#include <Network/NetworkResourceTreeTO.h>

#include <PersisterInterface/Definitions.h>
#include <PersisterInterface/DownloadCache.h>

#include "AlienWindow.h"
#include "BrowserData.h"
#include "Definitions.h"

class BrowserWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(BrowserWindow);

public:
    void onRefresh();
    void onPreviewPictureChanged(std::string const& resourceId);
    WorkspaceType getCurrentWorkspaceType() const;

    DownloadCache& getSimulationCache();

private:
    BrowserWindow();

    void initIntern() override;
    void shutdownIntern() override;

    void processIntern() override;
    void processBackground() override;
    void processActivated() override;

    void processToolbar();
    void processWorkspace();
    void processResourceView();
    bool isLoginRequired() const;
    void processWorkspaceSelection();
    void processFilter();
    void processUserList();
    void processStatusBar();

    void processEmojiWindow();
    void processEmojiButton(int emojiType);

    void processRefreshingScreen(RealVector2D const& startPos);

    void onEditResource(NetworkResourceTreeTO const& treeTO);
    void onReplaceResource(BrowserLeaf const& leaf);
    void onMoveResource(NetworkResourceTreeTO const& treeTO);
    void onDeleteResource(NetworkResourceTreeTO const& treeTO);
    void onExpandFolders();
    void onCollapseFolders();
    void openWeblink(std::string const& link);

    BrowserData _data;
    BrowserGalleryWidget _galleryWidget;
    BrowserTableWidget _tableWidget;
    BrowserUserListWidget _userListWidget;
    BrowserLoginHintWidget _loginHintWidget;

    bool _galleryView = true;
    bool _showAllEmojis = false;
    float _userTableWidth = 0;
};
