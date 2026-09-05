#pragma once

#include <chrono>
#include <optional>

#include <imgui.h>

#include <Base/Cache.h>
#include <Base/Hashes.h>

#include <Network/NetworkResourceRawTO.h>
#include <Network/NetworkResourceTreeTO.h>
#include <Network/UserTO.h>

#include <EngineInterface/Definitions.h>

#include <PersisterInterface/PersisterFacade.h>
#include <PersisterInterface/SerializerService.h>

#include "AlienWindow.h"
#include "Definitions.h"
#include "LastSessionBrowserData.h"

enum GallerySorting
{
    GallerySorting_MostReactions,
    GallerySorting_Newest,
    GallerySorting_MostDownloads
};

class BrowserWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(BrowserWindow);

public:
    void onRefresh();
    WorkspaceType getCurrentWorkspaceType() const;

    DownloadCache& getSimulationCache();

private:
    BrowserWindow();

    void initIntern() override;
    void shutdownIntern() override;

    struct WorkspaceId
    {
        NetworkResourceType resourceType;
        WorkspaceType workspaceType;
        auto operator<=>(WorkspaceId const&) const = default;
    };
    struct Workspace
    {
        std::vector<ImGuiTableColumnSortSpecs> sortSpecs;
        std::vector<NetworkResourceRawTO> rawTOs;    // Unfiltered, sorted
        std::vector<NetworkResourceTreeTO> treeTOs;  // Filtered, sorted
        std::set<std::vector<std::string>> collapsedFolderNames;
    };

    void refreshIntern(bool withRetry);

    void processIntern() override;
    void processBackground() override;

    void processToolbar();
    void processWorkspaceSelection();
    void processFilter();
    void processWorkspace();
    void processUserList();
    void processStatusBar();

    void processSimulationList();
    void processGenomeList();

    void processGallery();
    void processGallerySorting();
    void processGalleryPaging();
    std::string getGalleryPageText() const;
    float getGalleryPagerWidth() const;
    void processGalleryTile(NetworkResourceRawTO const& rawTO, float tileWidth);
    void processGalleryPicture(NetworkResourceRawTO const& rawTO, float width);
    std::vector<NetworkResourceRawTO> getSortedGalleryEntries() const;
    void requestMissingPictures(std::vector<NetworkResourceRawTO> const& pageEntries);
    bool hasPreviewPictures() const;

    bool processResourceNameField(
        NetworkResourceTreeTO const& treeTO,
        std::set<std::vector<std::string>>& collapsedFolderNames);  // Return true if folder symbol clicked
    void processDescriptionField(NetworkResourceTreeTO const& treeTO);
    void processReactionList(NetworkResourceTreeTO const& treeTO);
    void processTimestampField(NetworkResourceTreeTO const& treeTO);
    void processUserNameField(NetworkResourceTreeTO const& treeTO);
    void processNumDownloadsField(NetworkResourceTreeTO const& treeTO);
    void processWidthField(NetworkResourceTreeTO const& treeTO);
    void processHeightField(NetworkResourceTreeTO const& treeTO);
    void processNumObjectsField(NetworkResourceTreeTO const& treeTO, bool kobjects);
    void processSizeField(NetworkResourceTreeTO const& treeTO, bool kbyte);
    void processVersionField(NetworkResourceTreeTO const& treeTO);

    bool processFolderTreeSymbols(
        NetworkResourceTreeTO const& treeTO,
        std::set<std::vector<std::string>>& collapsedFolderNames);  // Return true if folder symbol clicked
    void processEmojiWindow();
    void processEmojiButton(int emojiType);

    void processDownloadButton(BrowserLeaf const& leaf);

    void processShortenedText(std::string const& text, bool bold = false);
    bool processActionButton(std::string const& text);
    bool processDetailButton();

    void processRefreshingScreen(RealVector2D const& startPos);

    void processActivated() override;

    void processPendingRequestIds();

    void createTreeTOs(Workspace& workspace);
    void sortUserList();

    void onSelectGalleryEntry(NetworkResourceRawTO const& rawTO);

    void onDownloadResource(BrowserLeaf const& leaf);
    void onReplaceResource(BrowserLeaf const& leaf);
    void onEditResource(NetworkResourceTreeTO const& treeTO);
    void onMoveResource(NetworkResourceTreeTO const& treeTO);
    void onDeleteResource(NetworkResourceTreeTO const& treeTO);
    void onToggleLike(NetworkResourceTreeTO const& to, int emojiType);
    void onExpandFolders();
    void onCollapseFolders();
    void openWeblink(std::string const& link);

    bool isOwner(NetworkResourceTreeTO const& treeTO) const;
    std::string getUserNamesToEmojiType(std::string const& resourceId, int emojiType);

    std::unordered_set<NetworkResourceRawTO> getAllRawTOs() const;

    void pushTextColor(NetworkResourceTreeTO const& to);
    void popTextColor();

    void drawOnlineSymbol();
    void drawLastDayOnlineSymbol();

    TaskProcessor _refreshProcessor;
    TaskProcessor _emojiUserNameProcessor;
    TaskProcessor _reactionProcessor;
    TaskProcessor _pictureProcessor;

    bool _galleryView = true;
    int _gallerySorting = GallerySorting_MostReactions;
    int _galleryPage = 0;
    int _galleryNumEntries = 0;
    int _galleryNumPages = 1;
    std::unordered_map<std::string, std::optional<TextureData>> _pictureBySimId;

    bool _activateEmojiPopup = false;
    bool _showAllEmojis = false;
    NetworkResourceTreeTO _emojiPopupTO;
    std::optional<std::chrono::steady_clock::time_point> _lastRefreshTime;

    std::vector<UserTO> _userTOs;
    WorkspaceId _currentWorkspace = {NetworkResourceType_Simulation, WorkspaceType_AlienProject};
    std::map<WorkspaceId, Workspace> _workspaces;
    LastSessionBrowserData _lastSessionData;

    NetworkResourceTreeTO _selectedTreeTO;

    std::string _filter;
    float _userTableWidth = 0;
    std::unordered_map<std::string, int> _ownEmojiTypeBySimId;
    std::unordered_map<std::pair<std::string, int>, std::set<std::string>> _userNamesByEmojiTypeBySimIdCache;

    std::vector<TextureData> _emojis;

    DownloadCache _downloadCache;

};
